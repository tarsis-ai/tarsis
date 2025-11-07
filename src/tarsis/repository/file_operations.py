"""
Helper functions for local git file operations.

This module provides utilities for advanced file operations that require
local git repository access, such as file renames with history preservation
and symlink creation.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

try:
    from git import Repo, GitCommandError
except ImportError:
    Repo = None
    GitCommandError = Exception

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Raised when a file operation fails."""
    pass


async def rename_file(
    old_path: str,
    new_path: str,
    repo: Repo,
    commit_message: Optional[str] = None
) -> str:
    """
    Rename file with git history preservation.

    Uses 'git mv' to ensure git blame, log, and history work correctly
    across the rename. This is far superior to delete+create which loses
    all file history.

    Args:
        old_path: Current file path (relative to repo root)
        new_path: New file path (relative to repo root)
        repo: GitPython Repo instance
        commit_message: Optional commit message (auto-generated if omitted)

    Returns:
        Commit SHA (hex string)

    Raises:
        FileOperationError: If old file doesn't exist or rename fails
        GitCommandError: If git operation fails

    Example:
        >>> repo = Repo("/path/to/repo")
        >>> sha = await rename_file("old.py", "new.py", repo)
        >>> # Git history preserved: git log --follow new.py works
    """
    if Repo is None:
        raise ImportError("GitPython is required for file operations")

    # Validate old file exists
    old_file = Path(repo.working_dir) / old_path
    if not old_file.exists():
        raise FileOperationError(f"Cannot rename: '{old_path}' does not exist")

    # Check if new path already exists
    new_file = Path(repo.working_dir) / new_path
    if new_file.exists():
        raise FileOperationError(
            f"Cannot rename: '{new_path}' already exists. "
            "Delete the existing file first or choose a different name."
        )

    # Create parent directory for new path if needed
    new_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Perform rename using git mv (preserves history)
        repo.git.mv(old_path, new_path)
        logger.info(f"Git rename: {old_path} → {new_path}")

        # Generate commit message if not provided
        if not commit_message:
            commit_message = f"Rename {old_path} to {new_path}"

        # Commit the rename
        commit = repo.index.commit(commit_message)
        logger.info(f"Committed rename: {commit.hexsha[:8]}")

        return commit.hexsha

    except GitCommandError as e:
        raise FileOperationError(f"Git rename failed: {e}")


async def create_symlink(
    link_path: str,
    target_path: str,
    repo: Repo,
    commit_message: Optional[str] = None
) -> str:
    """
    Create symbolic link and commit to repository.

    Creates a symlink and commits it to the git repository. Symlinks are
    stored natively in git with mode 120000.

    Platform Support:
    - Linux/macOS: Full native support
    - Windows: Requires Developer Mode or admin privileges

    Args:
        link_path: Path where symlink will be created (relative to repo root)
        target_path: Path symlink points to (can be relative or absolute)
        repo: GitPython Repo instance
        commit_message: Optional commit message

    Returns:
        Commit SHA (hex string)

    Raises:
        FileOperationError: If symlink creation fails or platform unsupported
        GitCommandError: If git operation fails

    Example:
        >>> repo = Repo("/path/to/repo")
        >>> sha = await create_symlink("link.txt", "target.txt", repo)
        >>> # Creates symlink stored in git as mode 120000
    """
    if Repo is None:
        raise ImportError("GitPython is required for file operations")

    # Check platform support
    support, error_msg = can_create_symlinks()
    if not support:
        raise FileOperationError(error_msg)

    # Create full paths
    repo_root = Path(repo.working_dir)
    link_full = repo_root / link_path

    # Check if link already exists
    if link_full.exists() or link_full.is_symlink():
        raise FileOperationError(
            f"Cannot create symlink: '{link_path}' already exists"
        )

    # Create parent directory if needed
    link_full.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Create symlink
        os.symlink(target_path, link_full)
        logger.info(f"Created symlink: {link_path} → {target_path}")

        # Stage symlink in git
        repo.index.add([link_path])

        # Generate commit message if not provided
        if not commit_message:
            commit_message = f"Add symlink {link_path} → {target_path}"

        # Commit
        commit = repo.index.commit(commit_message)
        logger.info(f"Committed symlink: {commit.hexsha[:8]}")

        return commit.hexsha

    except OSError as e:
        # Clean up if symlink creation failed
        if link_full.is_symlink():
            link_full.unlink()
        raise FileOperationError(f"Failed to create symlink: {e}")
    except GitCommandError as e:
        # Clean up if git operation failed
        if link_full.is_symlink():
            link_full.unlink()
        raise FileOperationError(f"Git operation failed: {e}")


def can_create_symlinks() -> tuple[bool, Optional[str]]:
    """
    Check if platform supports symlink creation.

    Tests whether the current platform and user permissions allow creating
    symbolic links. On Windows, this requires Developer Mode (Windows 10+)
    or administrator privileges.

    Returns:
        Tuple of (supported: bool, error_message: str | None)
        - (True, None) if symlinks are supported
        - (False, "error message") if not supported

    Example:
        >>> supported, error = can_create_symlinks()
        >>> if not supported:
        ...     print(f"Symlinks not available: {error}")
    """
    # Unix systems (Linux, macOS) always support symlinks
    if sys.platform != "win32":
        return True, None

    # Windows: Try to create a test symlink
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_link = Path(tmpdir) / "test_link"
            test_target = Path(tmpdir) / "test_target"
            test_target.touch()

            # Attempt to create symlink
            os.symlink(test_target, test_link)

            # Verify it worked
            if not test_link.is_symlink():
                return False, "Symlink creation appeared to succeed but link is invalid"

            # Clean up
            test_link.unlink()

            return True, None

    except (OSError, NotImplementedError) as e:
        error_msg = (
            "Symlinks are not supported on this Windows system. "
            "To enable symlinks:\n"
            "1. Open Settings\n"
            "2. Go to Update & Security > For developers\n"
            "3. Enable 'Developer Mode'\n"
            "4. Restart your terminal/IDE\n\n"
            "Alternatively, run as Administrator (not recommended)."
        )
        return False, error_msg


async def safe_push(repo: Repo, branch: str) -> None:
    """
    Push changes to remote with conflict detection.

    Attempts to push local commits to the remote branch. If the push is
    rejected because the remote has new commits (non-fast-forward), raises
    a clear error message.

    Args:
        repo: GitPython Repo instance
        branch: Branch name to push

    Raises:
        FileOperationError: If push rejected due to remote changes
        GitCommandError: If push fails for other reasons

    Example:
        >>> repo = Repo("/path/to/repo")
        >>> await safe_push(repo, "feature-branch")
    """
    if Repo is None:
        raise ImportError("GitPython is required for file operations")

    try:
        # Push to remote
        push_info = repo.remotes.origin.push(branch)
        logger.info(f"Pushed {branch} to origin")

        # Check push results
        for info in push_info:
            if info.flags & info.ERROR:
                raise FileOperationError(f"Push failed: {info.summary}")

    except GitCommandError as e:
        error_msg = str(e).lower()

        # Check if rejected due to non-fast-forward (remote has new commits)
        if "rejected" in error_msg or "non-fast-forward" in error_msg:
            raise FileOperationError(
                f"Cannot push to '{branch}': remote branch has new commits.\n"
                "The branch was modified by another user or process after you started.\n"
                "Please retry the operation with fresh changes."
            )

        # Check for other common errors
        if "no such ref" in error_msg or "does not exist" in error_msg:
            raise FileOperationError(
                f"Cannot push to '{branch}': branch does not exist on remote.\n"
                "The branch may need to be created first."
            )

        # Unknown push error
        raise FileOperationError(f"Push failed: {e}")


async def batch_modify_files(
    operations: List[Dict[str, Any]],
    repo: Repo,
    commit_message: str,
    auto_push: bool = True
) -> Dict[str, Any]:
    """
    Execute multiple file operations in a single atomic commit.

    Performs multiple file operations (create, update, delete, rename) and
    commits them all together in a single commit. This is much faster than
    individual commits for each file.

    Args:
        operations: List of operation dicts, each with:
            - type: "create" | "update" | "delete" | "rename"
            - path: file path (for create/update/delete)
            - content: file content (for create/update)
            - old_path: source path (for rename)
            - new_path: destination path (for rename)
        repo: GitPython Repo instance
        commit_message: Commit message for all changes
        auto_push: Whether to push after commit (default: True)

    Returns:
        Dict with:
            - commit_sha: The commit SHA
            - operations_count: Number of operations performed
            - files_modified: List of modified file paths
            - pushed: Whether changes were pushed

    Raises:
        FileOperationError: If any operation fails
        GitCommandError: If git operation fails

    Example:
        >>> operations = [
        ...     {"type": "create", "path": "new.py", "content": "# New file"},
        ...     {"type": "rename", "old_path": "a.py", "new_path": "b.py"},
        ...     {"type": "delete", "path": "old.py"},
        ... ]
        >>> result = await batch_modify_files(operations, repo, "Batch changes")
    """
    if Repo is None:
        raise ImportError("GitPython is required for file operations")

    if not operations:
        raise FileOperationError("No operations provided")

    repo_root = Path(repo.working_dir)
    modified_files = []

    try:
        # Execute all operations
        for i, op in enumerate(operations):
            op_type = op.get("type")

            if op_type == "create" or op_type == "update":
                # Create or update file
                path = op.get("path")
                content = op.get("content", "")

                if not path:
                    raise FileOperationError(f"Operation {i}: missing 'path'")

                file_path = repo_root / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                modified_files.append(path)
                logger.debug(f"{'Created' if op_type == 'create' else 'Updated'}: {path}")

            elif op_type == "delete":
                # Delete file
                path = op.get("path")

                if not path:
                    raise FileOperationError(f"Operation {i}: missing 'path'")

                file_path = repo_root / path
                if file_path.exists():
                    file_path.unlink()
                    modified_files.append(path)
                    logger.debug(f"Deleted: {path}")

            elif op_type == "rename":
                # Rename file using git mv
                old_path = op.get("old_path")
                new_path = op.get("new_path")

                if not old_path or not new_path:
                    raise FileOperationError(
                        f"Operation {i}: rename requires 'old_path' and 'new_path'"
                    )

                # Use git mv for history preservation
                repo.git.mv(old_path, new_path)
                modified_files.extend([old_path, new_path])
                logger.debug(f"Renamed: {old_path} → {new_path}")

            else:
                raise FileOperationError(
                    f"Operation {i}: unknown type '{op_type}'. "
                    "Supported: create, update, delete, rename"
                )

        # Stage all modified files
        # For git mv, files are already staged
        # For create/update/delete, we need to stage them
        files_to_stage = [f for f in modified_files if f]
        if files_to_stage:
            repo.index.add(files_to_stage)

        # Commit all changes
        commit = repo.index.commit(commit_message)
        logger.info(
            f"Batch commit: {len(operations)} operations → {commit.hexsha[:8]}"
        )

        # Push if requested
        pushed = False
        if auto_push:
            # Get current branch name
            branch = repo.active_branch.name
            await safe_push(repo, branch)
            pushed = True

        return {
            "commit_sha": commit.hexsha,
            "operations_count": len(operations),
            "files_modified": modified_files,
            "pushed": pushed
        }

    except Exception as e:
        # Rollback: reset to HEAD and clean working directory
        logger.error(f"Batch operation failed, rolling back: {e}")
        try:
            repo.git.reset("--hard", "HEAD")
            repo.git.clean("-fd")
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {rollback_error}")

        raise FileOperationError(f"Batch operation failed: {e}")
