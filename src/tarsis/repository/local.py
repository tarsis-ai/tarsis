"""
Local repository clone management.

This module provides the CloneManager class for managing local repository clones
with lifecycle management, cleanup, and branch operations.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

try:
    import git
    from git import Repo, GitCommandError
except ImportError:
    git = None
    Repo = None
    GitCommandError = Exception

logger = logging.getLogger(__name__)


class CloneError(Exception):
    """Raised when repository cloning fails."""
    pass


class CloneManager:
    """
    Manages local repository clones with lifecycle management.

    Provides methods to clone repositories, checkout branches, and cleanup
    resources. Supports both temporary and persistent workspace directories.

    Example:
        manager = CloneManager("myorg", "myrepo", "ghp_token123", task_id="abc-123")
        repo_path = await manager.ensure_clone(branch="feature-x")
        # Use repo_path for validation or code operations
        await manager.cleanup()  # Remove clone when done
    """

    def __init__(
        self,
        owner: str,
        name: str,
        token: str,
        task_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ):
        """
        Initialize clone manager for a repository.

        Args:
            owner: GitHub repository owner/organization
            name: Repository name
            token: GitHub personal access token
            task_id: Unique task identifier (for workspace isolation)
            workspace_dir: Custom workspace directory (overrides env/default)
        """
        if git is None:
            raise ImportError(
                "GitPython is required for local clone management. "
                "Install with: pip install GitPython"
            )

        self.owner = owner
        self.name = name
        self.token = token
        self.task_id = task_id or "default"

        # Determine workspace directory
        if workspace_dir:
            self._workspace_base = Path(workspace_dir)
        else:
            workspace_env = os.getenv("TARSIS_WORKSPACE_DIR")
            if workspace_env:
                self._workspace_base = Path(workspace_env)
            else:
                # Default: temporary directory
                self._workspace_base = Path(tempfile.gettempdir()) / "tarsis"

        # Create task-specific workspace
        self._workspace = self._workspace_base / f"task_{self.task_id}"
        self._clone_path: Optional[Path] = None
        self._repo: Optional[Repo] = None
        self._current_branch: Optional[str] = None

        logger.info(
            f"Initialized CloneManager for {owner}/{name} "
            f"(workspace: {self._workspace})"
        )

    @property
    def repo_url(self) -> str:
        """Get the HTTPS clone URL with authentication."""
        return f"https://{self.token}@github.com/{self.owner}/{self.name}.git"

    @property
    def repo_url_display(self) -> str:
        """Get the clone URL for display (without token)."""
        return f"https://github.com/{self.owner}/{self.name}.git"

    def is_cloned(self) -> bool:
        """Check if repository is currently cloned."""
        if not self._clone_path:
            return False
        if not self._clone_path.exists():
            return False
        if not (self._clone_path / ".git").exists():
            return False
        return True

    def get_repo_path(self) -> str:
        """
        Get path to local clone.

        Returns:
            Absolute path to cloned repository

        Raises:
            CloneError: If repository is not cloned
        """
        if not self.is_cloned():
            raise CloneError(
                f"Repository {self.owner}/{self.name} is not cloned. "
                "Call ensure_clone() first."
            )
        return str(self._clone_path)

    async def ensure_clone(
        self,
        branch: Optional[str] = None,
        shallow: bool = True,
        max_retries: int = 3,
    ) -> str:
        """
        Ensure repository is cloned and return path. Idempotent.

        If already cloned, verifies the clone is valid and checks out the
        requested branch if different. If not cloned, performs a fresh clone.

        Args:
            branch: Branch to checkout (default: repo default branch)
            shallow: If True, perform shallow clone (--depth=1) for speed
            max_retries: Maximum number of retry attempts on failure

        Returns:
            Absolute path to cloned repository

        Raises:
            CloneError: If cloning fails after all retries
        """
        # If already cloned, pull latest changes before returning
        if self.is_cloned():
            if branch and branch != self._current_branch:
                logger.info(f"Checking out branch: {branch}")
                await self.checkout_branch(branch)

            # Pull latest changes from remote
            logger.info(f"Pulling latest changes from remote for branch: {branch or self._current_branch}")
            try:
                await self.update(branch)
            except CloneError as e:
                logger.warning(f"Failed to pull latest changes: {e}. Using cached clone.")

            return str(self._clone_path)

        # Perform clone with retry logic
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Cloning {self.repo_url_display} "
                    f"(attempt {attempt}/{max_retries}, shallow={shallow})"
                )
                path = await self._clone_repository(branch, shallow)
                logger.info(f"Successfully cloned to: {path}")
                return path
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Clone attempt {attempt}/{max_retries} failed: {e}"
                )

                # Cleanup failed clone attempt
                if self._clone_path and self._clone_path.exists():
                    shutil.rmtree(self._clone_path, ignore_errors=True)
                self._clone_path = None
                self._repo = None

                # Wait before retry (exponential backoff)
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 2, 4, 8 seconds
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

        # All retries failed
        raise CloneError(
            f"Failed to clone {self.repo_url_display} after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def _clone_repository(
        self,
        branch: Optional[str],
        shallow: bool,
    ) -> str:
        """
        Perform the actual repository clone operation.

        Args:
            branch: Branch to checkout
            shallow: Whether to do shallow clone

        Returns:
            Path to cloned repository

        Raises:
            GitCommandError: If git clone fails
        """
        # Ensure workspace exists
        self._workspace.mkdir(parents=True, exist_ok=True)

        # Clone path
        self._clone_path = self._workspace / self.name

        # Remove if already exists (from failed attempt)
        if self._clone_path.exists():
            shutil.rmtree(self._clone_path, ignore_errors=True)

        # Prepare clone options
        clone_kwargs = {
            "depth": 1 if shallow else None,
            "branch": branch,
            "single_branch": shallow,
        }
        # Remove None values
        clone_kwargs = {k: v for k, v in clone_kwargs.items() if v is not None}

        # Run clone in thread pool (blocking operation)
        loop = asyncio.get_event_loop()
        self._repo = await loop.run_in_executor(
            None,
            lambda: Repo.clone_from(
                self.repo_url,
                str(self._clone_path),
                **clone_kwargs,
            ),
        )

        # Track current branch
        self._current_branch = branch or self._repo.active_branch.name

        return str(self._clone_path)

    async def checkout_branch(self, branch: str, create: bool = False) -> None:
        """
        Checkout a branch (optionally creating it).

        Args:
            branch: Branch name to checkout
            create: If True, create branch if it doesn't exist

        Raises:
            CloneError: If repository is not cloned
            GitCommandError: If checkout fails
        """
        if not self.is_cloned():
            raise CloneError("Cannot checkout branch: repository not cloned")

        logger.info(f"Checking out branch: {branch} (create={create})")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._checkout_branch_sync,
                branch,
                create,
            )
            self._current_branch = branch
            logger.info(f"Successfully checked out: {branch}")
        except GitCommandError as e:
            raise CloneError(f"Failed to checkout branch {branch}: {e}")

    def _checkout_branch_sync(self, branch: str, create: bool) -> None:
        """Synchronous branch checkout (called via executor)."""
        if create:
            # Create and checkout new branch
            self._repo.git.checkout("-b", branch)
        else:
            # Checkout existing branch
            self._repo.git.checkout(branch)

    async def update(self, branch: Optional[str] = None) -> None:
        """
        Pull latest changes from remote.

        Args:
            branch: Branch to update (default: current branch)

        Raises:
            CloneError: If repository is not cloned
            GitCommandError: If pull fails
        """
        if not self.is_cloned():
            raise CloneError("Cannot update: repository not cloned")

        target_branch = branch or self._current_branch
        logger.info(f"Pulling latest changes for branch: {target_branch}")

        try:
            # Checkout target branch if different
            if target_branch != self._current_branch:
                await self.checkout_branch(target_branch)

            # Pull changes
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._repo.remotes.origin.pull)
            logger.info("Successfully pulled latest changes")
        except GitCommandError as e:
            raise CloneError(f"Failed to pull changes: {e}")

    async def cleanup(self) -> None:
        """
        Remove local clone and cleanup resources.

        This method is idempotent and safe to call multiple times.
        Removes the entire task workspace directory.
        """
        cleanup_enabled = os.getenv("TARSIS_CLEANUP_ON_EXIT", "true").lower()
        if cleanup_enabled == "false":
            logger.info("Cleanup disabled via TARSIS_CLEANUP_ON_EXIT=false")
            return

        if not self._workspace.exists():
            logger.debug(f"Workspace already cleaned: {self._workspace}")
            return

        logger.info(f"Cleaning up workspace: {self._workspace}")
        try:
            shutil.rmtree(self._workspace, ignore_errors=False)
            logger.info("Workspace cleanup completed")
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace: {e}")
        finally:
            self._clone_path = None
            self._repo = None
            self._current_branch = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "cloned" if self.is_cloned() else "not cloned"
        branch_info = f", branch={self._current_branch}" if self._current_branch else ""
        return (
            f"CloneManager({self.owner}/{self.name}, "
            f"task_id={self.task_id}, {status}{branch_info})"
        )
