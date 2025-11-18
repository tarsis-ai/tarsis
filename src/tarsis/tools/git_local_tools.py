"""
Local git operation tool handlers.

These tools perform advanced file operations via local git repository access,
enabling features that are impossible or inefficient via the GitHub API:
- File rename with history preservation
- Symlink creation and management
- Batch file operations with atomic commits
- Branch creation and inspection
- Git diff operations
"""

import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

from .base import BaseToolHandler, ToolDefinition, ToolResponse, ToolCategory
from ..repository import (
    rename_file,
    create_symlink,
    can_create_symlinks,
    safe_push,
    batch_modify_files,
    FileOperationError
)
from ..commit.message_generator import (
    generate_commit_message,
    CommitContext,
    FileChange
)
from ..commit.grouping import (
    CommitGrouper,
    TypeBasedGrouping,
    DependencyAwareGrouping,
    SizeBasedGrouping,
    should_use_multi_commit
)

logger = logging.getLogger(__name__)


class RenameFileHandler(BaseToolHandler):
    """
    Tool to rename or move files while preserving git history.

    Uses 'git mv' to ensure git blame, log --follow, and history work correctly
    across renames. This is far superior to delete+create which loses all history.
    """

    @property
    def name(self) -> str:
        return "rename_file"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Rename or move a file while preserving complete git history.

**Why use this instead of delete+create:**
- Preserves file history: `git log --follow` works across the rename
- Maintains git blame: authorship information preserved
- Git recognizes as rename, not separate delete+add
- Much better for code archaeology and blame tracking

**Requirements:**
- Requires local repository clone (handled automatically)
- Works on all platforms

**When to use:**
- Refactoring code organization
- Moving files to different directories
- Renaming files for clarity

**Example:**
Rename src/old_name.py to src/new_name.py with full history preservation.

🤖 **AI-Powered Commit Messages**: Set `auto_generate_message: true` to automatically generate a conventional commit message. When enabled, `commit_message` can be omitted.""",
            input_schema={
                "type": "object",
                "properties": {
                    "old_path": {
                        "type": "string",
                        "description": "Current file path relative to repo root (e.g., 'src/old.py')"
                    },
                    "new_path": {
                        "type": "string",
                        "description": "New file path relative to repo root (e.g., 'src/new.py')"
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message. Optional if auto_generate_message is true. If omitted without auto-gen, defaults to 'Rename old_path to new_path'"
                    },
                    "auto_push": {
                        "type": "boolean",
                        "description": "Whether to automatically push changes to remote (default: true)",
                        "default": True
                    },
                    "auto_generate_message": {
                        "type": "boolean",
                        "description": "If true, automatically generate a conventional commit message using AI. Default: false",
                        "default": False
                    }
                },
                "required": ["old_path", "new_path"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute file rename operation."""

        try:
            # 1. Extract parameters
            old_path = input_data["old_path"]
            new_path = input_data["new_path"]
            commit_message = input_data.get("commit_message", "")
            auto_push = input_data.get("auto_push", True)
            auto_generate = input_data.get("auto_generate_message", False)

            # Generate commit message if requested
            if auto_generate and (not commit_message or commit_message.strip() == ""):
                llm_provider = getattr(context, 'llm_provider', None)
                if not llm_provider:
                    return self._error_response(
                        Exception(
                            "AI commit message generation requires LLM provider. "
                            "Cannot auto-generate without LLM access."
                        )
                    )

                try:
                    # Create context for rename operation
                    file_change = FileChange(
                        path=new_path,
                        change_type="rename",
                        old_path=old_path,
                        additions=0,
                        deletions=0
                    )

                    commit_context = CommitContext(
                        file_changes=[file_change],
                        branch_name=getattr(context, "branch_name", None)
                    )

                    logger.info(f"Generating commit message for rename: {old_path} → {new_path}")
                    generated = await generate_commit_message(
                        context=commit_context,
                        llm_provider=llm_provider,
                        temperature=0.3
                    )

                    commit_message = generated.message
                    logger.info(f"Generated commit message: {commit_message.split(chr(10))[0]}")

                except Exception as e:
                    logger.error(f"Failed to generate commit message: {e}")
                    return self._error_response(
                        Exception(f"Failed to generate commit message: {e}")
                    )

            # 2. Validate clone manager availability
            if not hasattr(context, "clone_manager") or not context.clone_manager:
                return self._error_response(
                    Exception(
                        "Clone manager not available. "
                        "Local git operations require CloneManager initialization."
                    )
                )

            # 3. Ensure repository is cloned
            # Note: Need full clone (not shallow) for complete git history
            branch = getattr(context, "branch_name", None)
            logger.info(f"Ensuring clone for rename operation (branch: {branch})")

            repo_path = await context.clone_manager.ensure_clone(
                branch=branch,
                shallow=False  # Full clone needed for git mv history
            )

            # 4. Get GitPython Repo instance
            repo = context.clone_manager._repo
            if not repo:
                return self._error_response(
                    Exception("Repository not available after clone")
                )

            # 5. Perform rename using helper function
            logger.info(f"Renaming {old_path} → {new_path}")
            commit_sha = await rename_file(
                old_path=old_path,
                new_path=new_path,
                repo=repo,
                commit_message=commit_message
            )

            # 6. Push if requested
            push_status = "committed locally (not pushed)"
            if auto_push and branch:
                try:
                    await safe_push(repo, branch)
                    push_status = "committed and pushed to remote"
                except FileOperationError as e:
                    # Commit succeeded but push failed
                    return self._error_response(
                        Exception(
                            f"File renamed and committed locally ({commit_sha[:8]}), "
                            f"but push failed: {e}\n\n"
                            f"The rename is saved in the local repository. "
                            f"You may need to resolve conflicts manually."
                        )
                    )

            # 7. Build success message
            success_message = f"""✅ File renamed successfully!

**Operation:** `{old_path}` → `{new_path}`
**Commit:** {commit_sha[:8]}
**Status:** {push_status}
**Git History:** Fully preserved

You can verify the rename preserved history with:
- `git log --follow {new_path}` - see complete file history across rename
- `git blame {new_path}` - see original authorship information"""

            return self._success_response(
                success_message,
                metadata={
                    "old_path": old_path,
                    "new_path": new_path,
                    "commit_sha": commit_sha,
                    "pushed": auto_push,
                    "history_preserved": True,
                    "operation": "rename"
                }
            )

        except FileOperationError as e:
            return self._error_response(Exception(f"Rename operation failed: {e}"))
        except Exception as e:
            logger.exception("Unexpected error in rename_file")
            return self._error_response(e)


class CreateSymlinkHandler(BaseToolHandler):
    """
    Tool to create symbolic links in the repository.

    Symlinks are stored natively in git and work across platforms.
    """

    @property
    def name(self) -> str:
        return "create_symlink"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Create a symbolic link in the repository.

**What is a symlink:**
A symlink (symbolic link) is a file that points to another file or directory.
It's stored natively in git and works across all platforms when the repo is cloned.

**Platform Support:**
- ✅ Linux: Full native support
- ✅ macOS: Full native support
- ⚠️  Windows: Requires Developer Mode (Windows 10+) or admin privileges

**When to use:**
- Link configuration files to canonical locations
- Create shortcuts to frequently accessed files
- Maintain backwards compatibility when moving files
- Share common files across directories

**Important Notes:**
- The target path can be relative or absolute
- If using relative paths, they're relative to the symlink location
- Git stores symlinks as mode 120000
- On Windows without Developer Mode, this will fail with a clear error message

**Example:**
Create a symlink `config/settings.json` → `../shared/default-settings.json`

🤖 **AI-Powered Commit Messages**: Set `auto_generate_message: true` to automatically generate a conventional commit message. When enabled, `commit_message` can be omitted.""",
            input_schema={
                "type": "object",
                "properties": {
                    "link_path": {
                        "type": "string",
                        "description": "Path where symlink will be created (relative to repo root, e.g., 'config/link.txt')"
                    },
                    "target_path": {
                        "type": "string",
                        "description": "Path the symlink points to. Can be relative (e.g., '../target.txt') or absolute"
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message. Optional if auto_generate_message is true. If omitted without auto-gen, defaults to 'Create symlink link_path → target_path'"
                    },
                    "auto_push": {
                        "type": "boolean",
                        "description": "Whether to automatically push changes to remote (default: true)",
                        "default": True
                    },
                    "auto_generate_message": {
                        "type": "boolean",
                        "description": "If true, automatically generate a conventional commit message using AI. Default: false",
                        "default": False
                    }
                },
                "required": ["link_path", "target_path"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute symlink creation."""

        try:
            # 1. Extract parameters
            link_path = input_data["link_path"]
            target_path = input_data["target_path"]
            commit_message = input_data.get("commit_message", "")
            auto_push = input_data.get("auto_push", True)
            auto_generate = input_data.get("auto_generate_message", False)

            # Generate commit message if requested
            if auto_generate and (not commit_message or commit_message.strip() == ""):
                llm_provider = getattr(context, 'llm_provider', None)
                if not llm_provider:
                    return self._error_response(
                        Exception(
                            "AI commit message generation requires LLM provider. "
                            "Cannot auto-generate without LLM access."
                        )
                    )

                try:
                    # Create context for symlink creation
                    file_change = FileChange(
                        path=link_path,
                        change_type="create",
                        additions=1,  # Symlink is essentially one line
                        deletions=0
                    )

                    commit_context = CommitContext(
                        file_changes=[file_change],
                        branch_name=getattr(context, "branch_name", None),
                        additional_context=f"Create symlink pointing to {target_path}"
                    )

                    logger.info(f"Generating commit message for symlink: {link_path} → {target_path}")
                    generated = await generate_commit_message(
                        context=commit_context,
                        llm_provider=llm_provider,
                        temperature=0.3
                    )

                    commit_message = generated.message
                    logger.info(f"Generated commit message: {commit_message.split(chr(10))[0]}")

                except Exception as e:
                    logger.error(f"Failed to generate commit message: {e}")
                    return self._error_response(
                        Exception(f"Failed to generate commit message: {e}")
                    )

            # 2. Check platform support first
            supported, error_msg = can_create_symlinks()
            if not supported:
                return self._error_response(
                    Exception(
                        f"Cannot create symlink: {error_msg}\n\n"
                        f"Symlink creation is not available on your system.\n"
                        f"Please enable the required features and try again."
                    )
                )

            # 3. Validate clone manager
            if not hasattr(context, "clone_manager") or not context.clone_manager:
                return self._error_response(
                    Exception(
                        "Clone manager not available. "
                        "Local git operations require CloneManager initialization."
                    )
                )

            # 4. Ensure repository is cloned
            branch = getattr(context, "branch_name", None)
            logger.info(f"Ensuring clone for symlink operation (branch: {branch})")

            repo_path = await context.clone_manager.ensure_clone(
                branch=branch,
                shallow=True  # Shallow clone is fine for symlinks
            )

            # 5. Get GitPython Repo instance
            repo = context.clone_manager._repo
            if not repo:
                return self._error_response(
                    Exception("Repository not available after clone")
                )

            # 6. Create symlink using helper function
            logger.info(f"Creating symlink {link_path} → {target_path}")
            commit_sha = await create_symlink(
                link_path=link_path,
                target_path=target_path,
                repo=repo,
                commit_message=commit_message
            )

            # 7. Push if requested
            push_status = "committed locally (not pushed)"
            if auto_push and branch:
                try:
                    await safe_push(repo, branch)
                    push_status = "committed and pushed to remote"
                except FileOperationError as e:
                    return self._error_response(
                        Exception(
                            f"Symlink created and committed locally ({commit_sha[:8]}), "
                            f"but push failed: {e}\n\n"
                            f"The symlink is saved in the local repository."
                        )
                    )

            # 8. Build success message
            success_message = f"""✅ Symlink created successfully!

**Link:** `{link_path}`
**Target:** `{target_path}`
**Commit:** {commit_sha[:8]}
**Status:** {push_status}
**Git Mode:** 120000 (symlink)

The symlink is stored in git and will work correctly when others clone the repository."""

            return self._success_response(
                success_message,
                metadata={
                    "link_path": link_path,
                    "target_path": target_path,
                    "commit_sha": commit_sha,
                    "pushed": auto_push,
                    "git_mode": "120000",
                    "operation": "create_symlink"
                }
            )

        except FileOperationError as e:
            return self._error_response(Exception(f"Symlink creation failed: {e}"))
        except Exception as e:
            logger.exception("Unexpected error in create_symlink")
            return self._error_response(e)


class ModifyFilesLocalHandler(BaseToolHandler):
    """
    Tool to modify multiple files via local git operations.

    Performs batch file operations (create, update, delete, rename) in a single
    atomic commit. Much faster than multiple GitHub API calls.
    """

    @property
    def name(self) -> str:
        return "modify_files_local"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Modify multiple files via local git. Supports both single-commit and multi-commit modes.

**Single-commit mode** (default): All operations in one commit
**Multi-commit mode**: Intelligently groups operations into logical commits by type

**Advantages over GitHub API (modify_file/commit_changes):**
- ⚡ 10-50x faster for bulk operations (no API rate limits)
- 📦 Atomic operations within each commit
- 🔄 Supports rename operations (with history preservation)
- 💾 Can handle binary files
- 🎯 Cleaner git history with logical grouping

**Supported Operations:**
1. **create** - Create new file
   - Required: path, content
2. **update** - Update existing file
   - Required: path, content
3. **delete** - Delete file
   - Required: path
4. **rename** - Rename/move file (preserves history)
   - Required: old_path, new_path

**When to use:**
- Modifying 5+ files at once
- Refactoring that touches many files
- When you need atomic behavior
- When rename history preservation is important

**Requirements:**
- Requires local repository clone (handled automatically)
- Operations are committed atomically (all-or-nothing)

**Example:**
Refactor a module by renaming files, updating imports, and deleting old code.

🤖 **AI-Powered Commit Messages**: Set `auto_generate_message: true` to automatically generate conventional commit messages. When enabled, `commit_message` can be omitted or empty.

📦 **Multi-commit Grouping**: Set `multi_commit: true` for large changesets (>5 files). Groups operations into logical commits:
- Separate commits for tests, docs, CI changes
- Ordered by dependencies (build → refactor → feat/fix → test → docs)
- Falls back to single commit when grouping adds no value""",
            input_schema={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "description": "List of file operations to perform",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["create", "update", "delete", "rename"],
                                    "description": "Type of operation to perform"
                                },
                                "path": {
                                    "type": "string",
                                    "description": "File path (for create/update/delete)"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "File content (for create/update operations)"
                                },
                                "old_path": {
                                    "type": "string",
                                    "description": "Source path (for rename operations)"
                                },
                                "new_path": {
                                    "type": "string",
                                    "description": "Destination path (for rename operations)"
                                }
                            },
                            "required": ["type"]
                        },
                        "minItems": 1
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message (required for single-commit, optional for multi-commit with auto_generate_message)"
                    },
                    "auto_push": {
                        "type": "boolean",
                        "description": "Whether to automatically push changes to remote (default: true)",
                        "default": True
                    },
                    "auto_generate_message": {
                        "type": "boolean",
                        "description": "If true, automatically generate conventional commit message(s) using AI. Default: false",
                        "default": False
                    },
                    "multi_commit": {
                        "type": "boolean",
                        "description": "If true, group operations into multiple logical commits. Recommended for >5 files. Default: false",
                        "default": False
                    },
                    "max_commits": {
                        "type": "integer",
                        "description": "Maximum commits to create in multi-commit mode. Default: 5",
                        "default": 5
                    }
                },
                "required": ["operations"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute batch file operations (single or multi-commit mode)."""

        try:
            # 1. Extract parameters
            operations = input_data["operations"]
            commit_message = input_data.get("commit_message", "")
            auto_push = input_data.get("auto_push", True)
            auto_generate = input_data.get("auto_generate_message", False)
            multi_commit = input_data.get("multi_commit", False)
            max_commits = input_data.get("max_commits", 5)

            if not operations:
                return self._error_response(
                    Exception("No operations provided. At least one operation is required.")
                )

            # Multi-commit mode: Group operations into logical commits
            if multi_commit:
                return await self._multi_commit_flow(
                    operations, auto_push, auto_generate, max_commits, context
                )

            # Single-commit mode (existing logic continues below)

            # Generate commit message if requested
            if auto_generate and (not commit_message or commit_message.strip() == ""):
                llm_provider = getattr(context, 'llm_provider', None)
                if not llm_provider:
                    return self._error_response(
                        Exception(
                            "AI commit message generation requires LLM provider. "
                            "Cannot auto-generate without LLM access."
                        )
                    )

                try:
                    # Create file changes context from operations
                    file_changes = []
                    for op in operations:
                        op_type = op.get("type")

                        if op_type == "rename":
                            file_change = FileChange(
                                path=op.get("new_path", ""),
                                change_type="rename",
                                old_path=op.get("old_path"),
                                additions=0,
                                deletions=0
                            )
                        else:
                            path = op.get("path", "")
                            content = op.get("content", "")
                            file_change = FileChange(
                                path=path,
                                change_type=op_type,
                                additions=len(content.split("\n")) if content and op_type in ["create", "update"] else 0,
                                deletions=0
                            )

                        file_changes.append(file_change)

                    commit_context = CommitContext(
                        file_changes=file_changes,
                        branch_name=getattr(context, "branch_name", None)
                    )

                    logger.info(f"Generating commit message for {len(operations)} operation(s)")
                    generated = await generate_commit_message(
                        context=commit_context,
                        llm_provider=llm_provider,
                        temperature=0.3
                    )

                    commit_message = generated.message
                    logger.info(f"Generated commit message: {commit_message.split(chr(10))[0]}")

                except Exception as e:
                    logger.error(f"Failed to generate commit message: {e}")
                    return self._error_response(
                        Exception(f"Failed to generate commit message: {e}")
                    )

            # Validate commit message is present
            if not commit_message or commit_message.strip() == "":
                return self._error_response(
                    Exception(
                        "commit_message is required. Either provide a message or set auto_generate_message=true."
                    )
                )

            # 2. Validate clone manager
            if not hasattr(context, "clone_manager") or not context.clone_manager:
                return self._error_response(
                    Exception(
                        "Clone manager not available. "
                        "Local git operations require CloneManager initialization."
                    )
                )

            # 3. Ensure repository is cloned
            branch = getattr(context, "branch_name", None)
            logger.info(f"Ensuring clone for batch operations (branch: {branch})")

            repo_path = await context.clone_manager.ensure_clone(
                branch=branch,
                shallow=False  # Need full clone if any renames
            )

            # 4. Get GitPython Repo instance
            repo = context.clone_manager._repo
            if not repo:
                return self._error_response(
                    Exception("Repository not available after clone")
                )

            # 5. Perform batch operations using helper function
            logger.info(f"Executing {len(operations)} file operations")
            result = await batch_modify_files(
                operations=operations,
                repo=repo,
                commit_message=commit_message,
                auto_push=False  # We'll handle push ourselves
            )

            # 6. Push if requested
            push_status = "committed locally (not pushed)"
            if auto_push and branch:
                try:
                    await safe_push(repo, branch)
                    push_status = "committed and pushed to remote"
                    result["pushed"] = True
                except FileOperationError as e:
                    return self._error_response(
                        Exception(
                            f"Batch operations committed locally ({result['commit_sha'][:8]}), "
                            f"but push failed: {e}\n\n"
                            f"Operations: {result['operations_count']}\n"
                            f"Files modified: {len(result['files_modified'])}"
                        )
                    )

            # 7. Build success message
            op_summary = self._build_operation_summary(operations)

            success_message = f"""✅ Batch file operations completed successfully!

**Operations performed:** {result['operations_count']}
**Files modified:** {len(result['files_modified'])}
**Commit:** {result['commit_sha'][:8]}
**Status:** {push_status}

**Operation Summary:**
{op_summary}

All changes have been committed atomically in a single commit."""

            return self._success_response(
                success_message,
                metadata={
                    "commit_sha": result["commit_sha"],
                    "operations_count": result["operations_count"],
                    "files_modified": result["files_modified"],
                    "pushed": result.get("pushed", False),
                    "operation": "batch_modify"
                }
            )

        except FileOperationError as e:
            return self._error_response(Exception(f"Batch operations failed: {e}"))
        except Exception as e:
            logger.exception("Unexpected error in modify_files_local")
            return self._error_response(e)

    async def _multi_commit_flow(
        self,
        operations: List[Dict[str, Any]],
        auto_push: bool,
        auto_generate: bool,
        max_commits: int,
        context: Any
    ) -> ToolResponse:
        """
        Execute multi-commit flow with intelligent grouping for local git operations.

        Groups operations into logical commits based on type, dependencies, and size.
        Falls back to single commit when grouping adds no value.

        Args:
            operations: List of file operations
            auto_push: Whether to push to remote
            auto_generate: Whether to auto-generate commit messages
            max_commits: Maximum number of commits to create
            context: Agent context (for LLM provider and clone manager)

        Returns:
            ToolResponse with commit details or error
        """
        try:
            # 1. Convert operations to FileChange objects
            file_changes = []
            for op in operations:
                op_type = op.get("type")

                if op_type == "rename":
                    file_change = FileChange(
                        path=op.get("new_path", ""),
                        change_type="rename",
                        old_path=op.get("old_path"),
                        additions=0,
                        deletions=0
                    )
                else:
                    path = op.get("path", "")
                    content = op.get("content", "")
                    file_change = FileChange(
                        path=path,
                        change_type=op_type,
                        additions=len(content.split("\n")) if content and op_type in ["create", "update"] else 0,
                        deletions=0
                    )

                file_changes.append(file_change)

            # 2. Check if multi-commit is beneficial
            if not should_use_multi_commit(file_changes, min_files=5):
                logger.info("Multi-commit adds no value, falling back to single commit")
                # Fall back to single commit
                input_data = {
                    "operations": operations,
                    "auto_push": auto_push,
                    "auto_generate_message": auto_generate
                }
                return await self.execute(input_data, context)

            # 3. Validate LLM provider for auto-generation
            if not auto_generate:
                return self._error_response(
                    Exception(
                        "Multi-commit mode requires auto_generate_message=true. "
                        "Commit messages for each group must be generated automatically."
                    )
                )

            llm_provider = getattr(context, 'llm_provider', None)
            if not llm_provider:
                return self._error_response(
                    Exception(
                        "AI commit message generation requires LLM provider. "
                        "Cannot use multi-commit mode without LLM access."
                    )
                )

            # 4. Validate clone manager
            if not hasattr(context, "clone_manager") or not context.clone_manager:
                return self._error_response(
                    Exception(
                        "Clone manager not available. "
                        "Local git operations require CloneManager initialization."
                    )
                )

            # 5. Ensure repository is cloned
            branch = getattr(context, "branch_name", None)
            logger.info(f"Ensuring clone for multi-commit operations (branch: {branch})")

            repo_path = await context.clone_manager.ensure_clone(
                branch=branch,
                shallow=False  # Need full clone for proper git operations
            )

            # 6. Get GitPython Repo instance
            repo = context.clone_manager._repo
            if not repo:
                return self._error_response(
                    Exception("Repository not available after clone")
                )

            # 7. Group operations using strategies
            logger.info(f"Grouping {len(file_changes)} operations into logical commits (max: {max_commits})")
            grouper = CommitGrouper(
                grouping_strategy=TypeBasedGrouping(min_files_per_group=2, merge_threshold=2),
                refinement_strategies=[
                    DependencyAwareGrouping(),
                    SizeBasedGrouping(max_files=15, max_loc=500)
                ],
                max_groups=max_commits
            )

            commit_groups = grouper.group_and_order(file_changes)
            logger.info(f"Created {len(commit_groups)} commit groups")

            # 8. If only 1 group resulted, fall back to single commit
            if len(commit_groups) == 1:
                logger.info("Grouping resulted in 1 group, using single commit")
                input_data = {
                    "operations": operations,
                    "auto_push": auto_push,
                    "auto_generate_message": True
                }
                return await self.execute(input_data, context)

            # 9. Create commits sequentially
            commit_shas = []
            commit_summaries = []

            for group_index, group in enumerate(commit_groups):
                logger.info(f"Creating commit {group_index + 1}/{len(commit_groups)}: {group.commit_type.value} ({group.file_count} operations)")

                # 9a. Generate commit message for this group
                commit_context = CommitContext(
                    file_changes=group.files,
                    branch_name=branch,
                    additional_context=f"Part {group_index + 1} of {len(commit_groups)}: {group.commit_type.value} changes"
                )

                try:
                    generated = await generate_commit_message(
                        context=commit_context,
                        llm_provider=llm_provider,
                        temperature=0.3
                    )
                    message = generated.message

                    # Add footer to indicate multi-commit sequence
                    message += f"\n\n---\nCommit {group_index + 1} of {len(commit_groups)} ({group.commit_type.value})"

                except Exception as e:
                    logger.error(f"Failed to generate commit message for group {group_index + 1}: {e}")
                    # Use fallback message
                    message = f"{group.commit_type.value}: {group.description_hint or 'update files'}\n\nCommit {group_index + 1} of {len(commit_groups)}"

                # 9b. Filter operations for this group
                group_operations = []
                for file_change in group.files:
                    # Find original operation from input
                    for op in operations:
                        # Match by path (handle rename specially)
                        if op.get("type") == "rename":
                            if op.get("new_path") == file_change.path:
                                group_operations.append(op)
                                break
                        else:
                            if op.get("path") == file_change.path:
                                group_operations.append(op)
                                break

                # 9c. Execute operations for this group using helper
                logger.info(f"Executing {len(group_operations)} operations for commit {group_index + 1}")
                result = await batch_modify_files(
                    operations=group_operations,
                    repo=repo,
                    commit_message=message,
                    auto_push=False  # We'll handle push ourselves at the end
                )

                commit_sha = result["commit_sha"]
                commit_shas.append(commit_sha)

                # 9d. Build summary for this commit
                commit_summaries.append({
                    "sha": commit_sha[:8],
                    "message": message.split("\n")[0],  # First line only
                    "type": group.commit_type.value,
                    "operations": len(group_operations),
                    "files": result["files_modified"],
                    "loc": group.total_loc
                })

                logger.info(f"Created commit {commit_sha[:8]}: {message.split(chr(10))[0]}")

            # 10. Push all commits if requested
            push_status = "committed locally (not pushed)"
            if auto_push and branch:
                try:
                    await safe_push(repo, branch)
                    push_status = "committed and pushed to remote"
                except FileOperationError as e:
                    return self._error_response(
                        Exception(
                            f"All {len(commit_shas)} commits created locally, but push failed: {e}\n\n"
                            f"Commits: {', '.join([s[:8] for s in commit_shas])}\n"
                            f"The commits are saved in the local repository."
                        )
                    )

            # 11. Build success message
            total_files = sum(len(s["files"]) for s in commit_summaries)
            result_msg = f"✅ Created {len(commit_shas)} commits with intelligent grouping!\n\n"
            result_msg += "**Commits Created:**\n"

            for i, summary in enumerate(commit_summaries, 1):
                result_msg += f"\n{i}. **{summary['sha']}** ({summary['type']})\n"
                result_msg += f"   {summary['message']}\n"
                result_msg += f"   Operations: {summary['operations']}, Files: {len(summary['files'])}, LOC: {summary['loc']}\n"
                result_msg += f"   Modified: {', '.join(summary['files'][:3])}"
                if len(summary['files']) > 3:
                    result_msg += f" (+{len(summary['files']) - 3} more)"
                result_msg += "\n"

            result_msg += f"\n**Status:** {push_status}\n"
            result_msg += f"**Total files modified:** {total_files}"

            # Collect all modified files from all commits
            all_files_modified = []
            for summary in commit_summaries:
                all_files_modified.extend(summary['files'])

            return self._success_response(
                result_msg,
                metadata={
                    "commit_count": len(commit_shas),
                    "commit_shas": commit_shas,
                    "commits": commit_summaries,
                    "total_operations": len(operations),
                    "total_files": total_files,
                    "files_modified": list(set(all_files_modified)),  # Deduplicate
                    "pushed": auto_push and push_status == "committed and pushed to remote",
                    "multi_commit": True,
                    "operation": "batch_modify_multi"
                }
            )

        except FileOperationError as e:
            logger.error(f"Multi-commit operation failed: {e}")
            return self._error_response(
                Exception(f"Multi-commit operation failed: {e}")
            )
        except Exception as e:
            logger.exception("Unexpected error in multi-commit flow")
            return self._error_response(e)

    def _build_operation_summary(self, operations: list) -> str:
        """Build a summary of operations for display."""
        summary_lines = []
        op_counts = {"create": 0, "update": 0, "delete": 0, "rename": 0}

        for op in operations:
            op_type = op.get("type", "unknown")
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        for op_type, count in op_counts.items():
            if count > 0:
                summary_lines.append(f"- {count} file(s) {op_type}d")

        return "\n".join(summary_lines) if summary_lines else "- No operations"


class CreateBranchLocalHandler(BaseToolHandler):
    """
    Tool to create a new branch via local git operations.

    Creates branches locally using git checkout -b, which is much faster than
    the GitHub API and works offline.
    """

    @property
    def name(self) -> str:
        return "create_branch_local"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Create a new branch via local git operations.

**Advantages over GitHub API:**
- ⚡ Faster (no network calls)
- 🔌 Works offline
- 🔄 More reliable (no rate limits)
- 🎯 Direct git operations

**How it works:**
1. Ensures repository is cloned locally
2. Creates branch from specified base (or current HEAD)
3. Optionally pushes branch to remote

**When to use:**
- Creating feature branches for implementation
- Creating hotfix branches
- Any branch creation that doesn't require immediate GitHub visibility
- Batch operations where speed matters

**Branch naming:**
- Use descriptive names (e.g., 'feature/add-auth', 'fix/memory-leak')
- Avoid special characters except dash and underscore
- Branch names are case-sensitive

**Example:**
Create branch 'feature/new-api' from 'main' and push to remote.""",
            input_schema={
                "type": "object",
                "properties": {
                    "branch_name": {
                        "type": "string",
                        "description": "Name of the branch to create (e.g., 'feature/add-login', 'fix/bug-123')"
                    },
                    "base_ref": {
                        "type": "string",
                        "description": "Base branch or commit SHA to create from (default: current HEAD). Examples: 'main', 'develop', or a commit SHA"
                    },
                    "push_to_remote": {
                        "type": "boolean",
                        "description": "Whether to push the new branch to remote after creation (default: false)",
                        "default": False
                    },
                    "force": {
                        "type": "boolean",
                        "description": "If true and branch exists, reset it to base_ref. If false and branch exists, return error (default: false)",
                        "default": False
                    }
                },
                "required": ["branch_name"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute local branch creation."""

        try:
            # 1. Extract parameters
            branch_name = input_data["branch_name"]
            base_ref = input_data.get("base_ref")
            push_to_remote = input_data.get("push_to_remote", False)
            force = input_data.get("force", False)

            # Validate branch name
            if not branch_name or not branch_name.strip():
                return self._error_response(
                    Exception("branch_name cannot be empty")
                )

            branch_name = branch_name.strip()

            # 2. Validate clone manager
            if not hasattr(context, "clone_manager") or not context.clone_manager:
                return self._error_response(
                    Exception(
                        "Clone manager not available. "
                        "Local git operations require CloneManager initialization."
                    )
                )

            # 3. Ensure repository is cloned
            logger.info(f"Ensuring clone for branch creation")
            repo_path = await context.clone_manager.ensure_clone(shallow=False)

            # 4. Get GitPython Repo instance
            repo = context.clone_manager._repo
            if not repo:
                return self._error_response(
                    Exception("Repository not available after clone")
                )

            # 5. Check if branch already exists
            branch_exists = False
            try:
                # Check local branches
                existing_branches = [ref.name for ref in repo.heads]
                branch_exists = branch_name in existing_branches
            except Exception as e:
                logger.warning(f"Failed to check existing branches: {e}")

            if branch_exists and not force:
                return self._error_response(
                    Exception(
                        f"Branch '{branch_name}' already exists locally. "
                        f"Use force=true to reset it to the base reference, "
                        f"or choose a different branch name."
                    )
                )

            # 6. Checkout base reference if specified
            if base_ref:
                logger.info(f"Checking out base reference: {base_ref}")
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        repo.git.checkout,
                        base_ref
                    )
                except Exception as e:
                    return self._error_response(
                        Exception(
                            f"Failed to checkout base reference '{base_ref}': {e}. "
                            f"Ensure the reference exists in the repository."
                        )
                    )

            # 7. Get the current commit SHA (will be the base of new branch)
            base_commit_sha = repo.head.commit.hexsha

            # 8. Create or reset branch
            logger.info(f"Creating branch '{branch_name}' from {base_commit_sha[:8]}")
            try:
                loop = asyncio.get_event_loop()
                if branch_exists and force:
                    # Reset existing branch to base
                    await loop.run_in_executor(
                        None,
                        repo.git.checkout,
                        "-B",  # Create or reset branch
                        branch_name
                    )
                    action = "reset"
                else:
                    # Create new branch
                    await loop.run_in_executor(
                        None,
                        repo.git.checkout,
                        "-b",
                        branch_name
                    )
                    action = "created"
            except Exception as e:
                return self._error_response(
                    Exception(f"Failed to create branch: {e}")
                )

            # Update clone manager's current branch tracker
            context.clone_manager._current_branch = branch_name

            # 9. Push to remote if requested
            push_status = "local only (not pushed)"
            if push_to_remote:
                try:
                    logger.info(f"Pushing branch '{branch_name}' to remote")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: repo.git.push(
                            "origin",
                            branch_name,
                            "--set-upstream"
                        )
                    )
                    push_status = "pushed to remote"
                except Exception as e:
                    return self._error_response(
                        Exception(
                            f"Branch '{branch_name}' created locally but push failed: {e}\n\n"
                            f"The branch exists in your local repository at {base_commit_sha[:8]}. "
                            f"You can push it manually later with: git push origin {branch_name}"
                        )
                    )

            # 10. Build success message
            success_message = f"""✅ Branch {action} successfully!

**Branch:** `{branch_name}`
**Base commit:** {base_commit_sha[:8]}
**Status:** {push_status}

You can now use this branch for your changes."""

            if base_ref:
                success_message += f"\n**Created from:** `{base_ref}`"

            return self._success_response(
                success_message,
                metadata={
                    "branch_name": branch_name,
                    "base_commit": base_commit_sha,
                    "base_ref": base_ref,
                    "pushed": push_to_remote,
                    "action": action,
                    "operation": "create_branch_local"
                }
            )

        except Exception as e:
            logger.exception("Unexpected error in create_branch_local")
            return self._error_response(e)


class GetBranchesLocalHandler(BaseToolHandler):
    """
    Tool to list and inspect branches via local git operations.

    Provides information about local and remote branches, including the current
    active branch and tracking information.
    """

    @property
    def name(self) -> str:
        return "get_branches_local"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""List and inspect branches via local git operations.

**Information provided:**
- 📍 Current active branch
- 📋 All local branches
- 🌐 Remote branches (from origin)
- 🔗 Tracking relationships between local and remote branches
- 📊 Latest commit SHA for each branch

**When to use:**
- Verify which branch you're currently on
- List available branches before switching
- Check if a branch exists before creating
- Understand branch tracking relationships
- Inspect branch state before operations

**Advantages over GitHub API:**
- ⚡ Faster (no network calls for local branches)
- 📦 Shows both local and remote branches in one call
- 🎯 Includes tracking information

**Example:**
Check which branch you're on and what other branches are available.""",
            input_schema={
                "type": "object",
                "properties": {
                    "include_remote": {
                        "type": "boolean",
                        "description": "Whether to include remote branches from origin (default: true)",
                        "default": True
                    }
                },
                "required": []
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute branch listing."""

        try:
            # 1. Extract parameters
            include_remote = input_data.get("include_remote", True)

            # 2. Validate clone manager
            if not hasattr(context, "clone_manager") or not context.clone_manager:
                return self._error_response(
                    Exception(
                        "Clone manager not available. "
                        "Local git operations require CloneManager initialization."
                    )
                )

            # 3. Ensure repository is cloned
            logger.info("Ensuring clone for branch inspection")
            repo_path = await context.clone_manager.ensure_clone(shallow=False)

            # 4. Get GitPython Repo instance
            repo = context.clone_manager._repo
            if not repo:
                return self._error_response(
                    Exception("Repository not available after clone")
                )

            # 5. Get current branch
            try:
                current_branch = repo.active_branch.name
            except Exception as e:
                # Might be in detached HEAD state
                current_branch = f"(detached at {repo.head.commit.hexsha[:8]})"
                logger.warning(f"Not on a branch: {e}")

            # 6. Get local branches
            local_branches = []
            for head in repo.heads:
                branch_info = {
                    "name": head.name,
                    "commit": head.commit.hexsha,
                    "is_current": head.name == current_branch
                }

                # Add tracking information if available
                try:
                    tracking = head.tracking_branch()
                    if tracking:
                        branch_info["tracking"] = tracking.name
                except Exception:
                    pass

                local_branches.append(branch_info)

            # 7. Get remote branches if requested
            remote_branches = []
            if include_remote:
                try:
                    for ref in repo.remotes.origin.refs:
                        # Skip HEAD reference
                        if ref.name == "origin/HEAD":
                            continue
                        remote_branches.append({
                            "name": ref.name,
                            "commit": ref.commit.hexsha
                        })
                except Exception as e:
                    logger.warning(f"Failed to get remote branches: {e}")

            # 8. Build success message
            success_message = f"""✅ Branch information retrieved!

**Current branch:** `{current_branch}`

**Local branches:** ({len(local_branches)} total)"""

            for branch in local_branches:
                marker = " ← current" if branch.get("is_current") else ""
                tracking_info = f" → tracks {branch['tracking']}" if branch.get("tracking") else ""
                success_message += f"\n- `{branch['name']}` @ {branch['commit'][:8]}{tracking_info}{marker}"

            if include_remote and remote_branches:
                success_message += f"\n\n**Remote branches:** ({len(remote_branches)} total)"
                for branch in remote_branches[:10]:  # Limit to first 10 for readability
                    success_message += f"\n- `{branch['name']}` @ {branch['commit'][:8]}"
                if len(remote_branches) > 10:
                    success_message += f"\n- ... and {len(remote_branches) - 10} more"

            return self._success_response(
                success_message,
                metadata={
                    "current_branch": current_branch,
                    "local_branches": local_branches,
                    "remote_branches": remote_branches if include_remote else [],
                    "operation": "get_branches_local"
                }
            )

        except Exception as e:
            logger.exception("Unexpected error in get_branches_local")
            return self._error_response(e)


class GetDiffLocalHandler(BaseToolHandler):
    """
    Tool to get git diff via local git operations.

    Shows differences between commits, branches, or the working directory,
    useful for code review and understanding changes.
    """

    @property
    def name(self) -> str:
        return "get_diff_local"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GIT

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Get git diff via local git operations.

**Diff modes:**
1. **Between two refs** - Compare any two commits/branches/tags
   - Set both `ref1` and `ref2` (e.g., 'main' and 'feature-branch')
2. **Working directory vs HEAD** - Show unstaged changes
   - Leave both `ref1` and `ref2` empty
3. **Staged changes** - Show what would be committed
   - Set `staged=true`
4. **Single ref vs working directory** - Changes since a commit
   - Set only `ref1`

**When to use:**
- Review changes before committing
- Compare branches before merging
- Understand what changed between commits
- Verify modifications during implementation
- Generate patches for review

**Output format:**
- Unified diff format (standard git diff)
- Shows file paths, line numbers, additions (+), deletions (-)
- Color-coded in supported terminals

**Examples:**
- Compare branches: `ref1="main"`, `ref2="feature-x"`
- Show unstaged changes: (no parameters)
- Show staged changes: `staged=true`
- Changes since commit: `ref1="abc123"`""",
            input_schema={
                "type": "object",
                "properties": {
                    "ref1": {
                        "type": "string",
                        "description": "First reference (branch, commit, tag). If only ref1 provided, compares ref1 to working directory"
                    },
                    "ref2": {
                        "type": "string",
                        "description": "Second reference (branch, commit, tag). Compares ref1 to ref2"
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "If true, show staged changes only (git diff --staged). Ignores ref1/ref2 (default: false)",
                        "default": False
                    },
                    "paths": {
                        "type": "array",
                        "description": "Optional list of file paths to limit diff to specific files",
                        "items": {"type": "string"}
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines to show around changes (default: 3)",
                        "default": 3
                    }
                },
                "required": []
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute git diff operation."""

        try:
            # 1. Extract parameters
            ref1 = input_data.get("ref1")
            ref2 = input_data.get("ref2")
            staged = input_data.get("staged", False)
            paths = input_data.get("paths", [])
            context_lines = input_data.get("context_lines", 3)

            # 2. Validate clone manager
            if not hasattr(context, "clone_manager") or not context.clone_manager:
                return self._error_response(
                    Exception(
                        "Clone manager not available. "
                        "Local git operations require CloneManager initialization."
                    )
                )

            # 3. Ensure repository is cloned
            logger.info("Ensuring clone for diff operation")
            repo_path = await context.clone_manager.ensure_clone(shallow=False)

            # 4. Get GitPython Repo instance
            repo = context.clone_manager._repo
            if not repo:
                return self._error_response(
                    Exception("Repository not available after clone")
                )

            # 5. Build diff command
            diff_args = [f"-U{context_lines}"]

            if staged:
                # Staged changes
                diff_args.append("--staged")
                diff_description = "staged changes"
            elif ref1 and ref2:
                # Between two refs
                diff_args.append(f"{ref1}..{ref2}")
                diff_description = f"{ref1} → {ref2}"
            elif ref1:
                # Single ref vs working directory
                diff_args.append(ref1)
                diff_description = f"{ref1} → working directory"
            else:
                # Working directory vs HEAD (unstaged changes)
                diff_description = "unstaged changes (working directory vs HEAD)"

            # Add paths if specified
            if paths:
                diff_args.append("--")
                diff_args.extend(paths)

            # 6. Execute diff
            logger.info(f"Getting diff: {diff_description}")
            try:
                loop = asyncio.get_event_loop()
                diff_output = await loop.run_in_executor(
                    None,
                    lambda: repo.git.diff(*diff_args)
                )
            except Exception as e:
                return self._error_response(
                    Exception(f"Failed to get diff: {e}")
                )

            # 7. Handle empty diff
            if not diff_output or diff_output.strip() == "":
                return self._success_response(
                    f"No differences found for: {diff_description}",
                    metadata={
                        "diff": "",
                        "ref1": ref1,
                        "ref2": ref2,
                        "staged": staged,
                        "has_changes": False,
                        "operation": "get_diff_local"
                    }
                )

            # 8. Build success message with diff
            # Truncate very large diffs for display
            max_display_lines = 500
            diff_lines = diff_output.split("\n")
            truncated = len(diff_lines) > max_display_lines

            if truncated:
                display_diff = "\n".join(diff_lines[:max_display_lines])
                display_diff += f"\n\n... (truncated {len(diff_lines) - max_display_lines} lines)"
            else:
                display_diff = diff_output

            success_message = f"""✅ Diff retrieved successfully!

**Comparing:** {diff_description}
**Context lines:** {context_lines}
**Total lines:** {len(diff_lines)}

**Diff:**
```diff
{display_diff}
```"""

            return self._success_response(
                success_message,
                metadata={
                    "diff": diff_output,  # Full diff in metadata
                    "ref1": ref1,
                    "ref2": ref2,
                    "staged": staged,
                    "paths": paths,
                    "line_count": len(diff_lines),
                    "truncated": truncated,
                    "has_changes": True,
                    "operation": "get_diff_local"
                }
            )

        except Exception as e:
            logger.exception("Unexpected error in get_diff_local")
            return self._error_response(e)
