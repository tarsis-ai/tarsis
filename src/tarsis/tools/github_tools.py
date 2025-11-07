"""
GitHub API tool handlers.

These tools allow the agent to interact with GitHub:
- Read issues
- Create branches
- Modify files (create/update/delete)
- Commit changes (multi-file)
- Create pull requests
- Post comments
"""

import os
import logging
from typing import Dict, Any, Optional, List
from .base import BaseToolHandler, ToolDefinition, ToolResponse, ToolCategory
from ..github import GitHubClient, GitHubConfig
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


# Global GitHub client instance
_github_client: Optional[GitHubClient] = None


def _get_github_client() -> GitHubClient:
    """Get or create the GitHub client instance."""
    global _github_client

    if _github_client is None:
        config = GitHubConfig(
            token=os.getenv("GITHUB_TOKEN", ""),
            repo_owner=os.getenv("GITHUB_REPO_OWNER", ""),
            repo_name=os.getenv("GITHUB_REPO_NAME", "")
        )
        _github_client = GitHubClient(config)

    return _github_client


class ReadIssueHandler(BaseToolHandler):
    """Tool to read GitHub issue details and comments"""

    @property
    def name(self) -> str:
        return "read_issue"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GITHUB

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Read a GitHub issue's title, description, and comments. Use this to understand what needs to be implemented.",
            input_schema={
                "type": "object",
                "properties": {
                    "issue_number": {
                        "type": "integer",
                        "description": "The GitHub issue number to read"
                    }
                },
                "required": ["issue_number"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Read issue details"""
        github = _get_github_client()
        await github.connect()

        try:
            issue_number = input_data["issue_number"]

            # Fetch issue details and comments
            issue = await github.get_issue(issue_number)
            comments = await github.get_issue_comments(issue_number)

            # Format response
            result = f"""# Issue #{issue_number}: {issue.title}

## Description
{issue.body}

## Comments
"""
            if comments:
                for i, comment in enumerate(comments, 1):
                    result += f"\n### Comment {i}\n{comment}\n"
            else:
                result += "\nNo comments yet."

            return self._success_response(result)

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()


class CreateBranchHandler(BaseToolHandler):
    """Tool to create a new Git branch"""

    @property
    def name(self) -> str:
        return "create_branch"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GITHUB

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Create a new branch for implementing the issue. The branch will be based on the default branch. If the branch already exists with no commits, it will be reused. If it has commits (from another user or previous work), you'll get an error and must use a different branch name.",
            input_schema={
                "type": "object",
                "properties": {
                    "branch_name": {
                        "type": "string",
                        "description": "Name for the new branch (e.g., 'feat/add-login-page')"
                    }
                },
                "required": ["branch_name"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Create a new branch"""
        github = _get_github_client()
        await github.connect()

        try:
            branch_name = input_data["branch_name"]

            # Get default branch SHA
            default_branch = await github.get_default_branch()
            base_sha = await github.get_branch_sha(default_branch)

            # Check if branch already exists
            try:
                existing_sha = await github.get_branch_sha(branch_name)

                # Branch exists - check if it's safe to reuse
                # Safe = branch points to the same commit as base (no new commits)
                if existing_sha == base_sha:
                    # Branch exists but has no commits beyond base - safe to reuse
                    return self._success_response(
                        f"Branch '{branch_name}' already exists and has no commits (reusing empty branch)",
                        metadata={
                            "branch_name": branch_name,
                            "base_branch": default_branch,
                            "already_existed": True,
                            "safe_to_reuse": True
                        }
                    )
                else:
                    # Branch has diverged - NOT safe to reuse (might have someone else's work)
                    return self._error_response(
                        Exception(
                            f"Branch '{branch_name}' already exists and has commits. "
                            f"This branch may contain work from another user or a previous implementation. "
                            f"Please use a different branch name (e.g., '{branch_name}-v2') to avoid conflicts."
                        )
                    )
            except Exception:
                # Branch doesn't exist, proceed with creation
                pass

            # Create branch
            await github.create_branch(branch_name, base_sha)

            return self._success_response(
                f"Successfully created branch '{branch_name}' from '{default_branch}'",
                metadata={
                    "branch_name": branch_name,
                    "base_branch": default_branch,
                    "already_existed": False
                }
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()


class CreatePullRequestHandler(BaseToolHandler):
    """Tool to create a pull request"""

    @property
    def name(self) -> str:
        return "create_pull_request"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GITHUB

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Create a pull request with your implementation. Use this after you've committed all changes.

⚠️ **IMPORTANT**: You MUST run `run_validation` before calling this tool to ensure code quality and catch errors (syntax errors, import issues, test failures, etc.). Creating a PR without validation wastes reviewer time and delays merges.""",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Pull request title"
                    },
                    "body": {
                        "type": "string",
                        "description": "Pull request description (supports Markdown)"
                    },
                    "head_branch": {
                        "type": "string",
                        "description": "The branch with your changes"
                    },
                    "base_branch": {
                        "type": "string",
                        "description": "The branch to merge into (usually 'main' or 'master')",
                        "default": "main"
                    }
                },
                "required": ["title", "body", "head_branch"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Create a pull request"""
        # Check if validation was performed (context is AgentTask instance)
        validation_performed = getattr(context, 'validation_performed', False)
        validation_passed = getattr(context, 'validation_passed', False)

        if not validation_performed:
            # Validation not performed - return warning
            warning_msg = """⚠️ **WARNING: No Validation Performed**

You are attempting to create a pull request without running validation first.

**Why validation is critical:**
- Catches syntax errors (like incomplete code: `print(` instead of `print('Hello')`)
- Detects import/dependency issues
- Runs tests to ensure functionality works
- Checks code quality via linting

**What you should do:**
1. Use the `run_validation` tool to check your changes
2. Fix any errors found
3. Then create the pull request

**Do you want to proceed anyway?**
If you're certain the code is correct and this is a special case (e.g., docs-only change),
you can proceed. However, it's **strongly recommended** to validate first.

To proceed: Call this tool again after running validation."""

            return self._success_response(
                warning_msg,
                metadata={
                    "validation_performed": False,
                    "warning": True,
                    "recommended_action": "run_validation"
                }
            )

        github = _get_github_client()
        await github.connect()

        try:
            title = input_data["title"]
            body = input_data["body"]
            head_branch = input_data["head_branch"]
            base_branch = input_data.get("base_branch") or await github.get_default_branch()

            # Add validation status to PR body
            if validation_passed:
                body += "\n\n---\n✅ **Validation:** Passed\n"
            else:
                body += "\n\n---\n⚠️ **Validation:** Did not pass all checks (see logs for details)\n"

            # Create PR
            pr = await github.create_pull_request(title, body, head_branch, base_branch)

            return self._success_response(
                f"Successfully created pull request: {pr.html_url}",
                metadata={
                    "pr_number": pr.number,
                    "pr_url": pr.html_url
                }
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()


class PostCommentHandler(BaseToolHandler):
    """Tool to post a comment on an issue"""

    @property
    def name(self) -> str:
        return "post_comment"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GITHUB

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Post a comment on the GitHub issue. Use this to provide status updates or ask for clarification.",
            input_schema={
                "type": "object",
                "properties": {
                    "issue_number": {
                        "type": "integer",
                        "description": "The issue number to comment on"
                    },
                    "comment": {
                        "type": "string",
                        "description": "The comment text (supports Markdown)"
                    }
                },
                "required": ["issue_number", "comment"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Post a comment"""
        github = _get_github_client()
        await github.connect()

        try:
            issue_number = input_data["issue_number"]
            comment = input_data["comment"]

            await github.post_issue_comment(issue_number, comment)

            return self._success_response(
                f"Successfully posted comment to issue #{issue_number}"
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()


class ModifyFileHandler(BaseToolHandler):
    """Tool to create or update a file and commit it to a branch"""

    @property
    def name(self) -> str:
        return "modify_file"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GITHUB

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Create or update a single file on a branch and commit the change. Use this to implement code changes, create new files, or update existing ones.

💡 **Reminder**: After modifying files, use `run_validation` to check for errors before creating a pull request.

🤖 **AI-Powered Commit Messages**: Set `auto_generate_message: true` to automatically generate a conventional commit message based on the file changes. When enabled, `commit_message` can be omitted or empty.""",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file relative to repository root (e.g., 'src/main.py')"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content (not a diff - the full file)"
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch to commit to (must already exist)"
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message describing the change. Optional if auto_generate_message is true."
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["create", "update", "auto"],
                        "description": "Operation type: 'create' (file must not exist), 'update' (file must exist), 'auto' (detect automatically). Default: auto",
                        "default": "auto"
                    },
                    "auto_generate_message": {
                        "type": "boolean",
                        "description": "If true, automatically generate a conventional commit message using AI. When enabled, commit_message can be omitted. Default: false",
                        "default": False
                    }
                },
                "required": ["file_path", "content", "branch"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Modify a file and commit"""
        github = _get_github_client()
        await github.connect()

        try:
            file_path = input_data["file_path"]
            content = input_data["content"]
            branch = input_data["branch"]
            commit_message = input_data.get("commit_message", "")
            operation = input_data.get("operation", "auto")
            auto_generate = input_data.get("auto_generate_message", False)

            # Generate commit message if requested
            if auto_generate and (not commit_message or commit_message.strip() == ""):
                # Check if we have LLM provider in context
                llm_provider = getattr(context, 'llm_provider', None)
                if not llm_provider:
                    return self._error_response(
                        Exception(
                            "AI commit message generation requires LLM provider. "
                            "Cannot auto-generate without LLM access."
                        )
                    )

                try:
                    # Determine operation type first for better message generation
                    branch_sha = await github.get_branch_sha(branch)
                    file_exists = await github.get_file_content(file_path, ref=branch) is not None
                    actual_operation = "update" if file_exists else "create"

                    # Create file change context
                    file_change = FileChange(
                        path=file_path,
                        change_type=actual_operation,
                        additions=len(content.split("\n")),
                        deletions=0,
                        diff_snippet=None
                    )

                    commit_context = CommitContext(
                        file_changes=[file_change],
                        branch_name=branch
                    )

                    # Generate message
                    logger.info(f"Generating commit message for {file_path} ({actual_operation})")
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

            # Get branch HEAD
            try:
                branch_sha = await github.get_branch_sha(branch)
            except Exception as e:
                return self._error_response(
                    Exception(f"Branch '{branch}' not found. Create it first using create_branch tool.")
                )

            # Get current tree
            tree_data = await github.get_git_tree(branch_sha, recursive=False)
            base_tree_sha = tree_data["sha"]

            # Check if file exists (for operation validation)
            file_exists = await github.get_file_content(file_path, ref=branch) is not None

            # Validate operation
            if operation == "create" and file_exists:
                return self._error_response(
                    Exception(f"File '{file_path}' already exists on branch '{branch}'. Use operation='update' or 'auto'.")
                )
            elif operation == "update" and not file_exists:
                return self._error_response(
                    Exception(f"File '{file_path}' does not exist on branch '{branch}'. Use operation='create' or 'auto'.")
                )

            # Determine actual operation
            actual_operation = "update" if file_exists else "create"

            # Create blob for file content
            blob_sha = await github.create_blob(content)

            # Create tree with the file change
            file_changes = [{
                "path": file_path,
                "mode": "100644",  # Regular file
                "type": "blob",
                "sha": blob_sha
            }]

            new_tree_sha = await github.create_tree(base_tree_sha, file_changes)

            # Create commit
            new_commit_sha = await github.create_commit(
                tree_sha=new_tree_sha,
                parent_sha=branch_sha,
                message=commit_message
            )

            # Update branch ref
            await github.update_branch_ref(branch, new_commit_sha)

            # Build success message
            action = "Created" if actual_operation == "create" else "Updated"
            result = f"{action} file '{file_path}' on branch '{branch}'\n"
            result += f"Commit: {new_commit_sha[:8]}\n"
            result += f"Message: {commit_message}\n\n"
            result += "💡 **Next step**: Run `run_validation` to check for errors before creating a pull request."

            return self._success_response(
                result,
                metadata={
                    "file_path": file_path,
                    "branch": branch,
                    "commit_sha": new_commit_sha,
                    "operation": actual_operation,
                    "file_size": len(content)
                }
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()


class CommitChangesHandler(BaseToolHandler):
    """Tool to commit multiple file changes atomically"""

    @property
    def name(self) -> str:
        return "commit_changes"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.GITHUB

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Commit multiple file changes (create/update/delete) atomically. Supports both single-commit and multi-commit modes.

**Single-commit mode** (default): All changes in one commit
**Multi-commit mode**: Intelligently groups changes into logical commits by type (feat/fix/test/docs)

💡 **Reminder**: After committing changes, use `run_validation` to check for errors before creating a pull request.

🤖 **AI-Powered Commit Messages**: Set `auto_generate_message: true` to automatically generate conventional commit messages.

📦 **Multi-commit Grouping**: Set `multi_commit: true` for large changesets (>5 files). Groups changes into logical commits:
- Separate commits for tests, docs, CI changes
- Ordered by dependencies (build → refactor → feat/fix → test → docs)
- Falls back to single commit when grouping adds no value""",
            input_schema={
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch to commit to (must already exist)"
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message (required for single-commit, optional for multi-commit with auto_generate_message)"
                    },
                    "files": {
                        "type": "array",
                        "description": "List of file changes to commit",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "File path relative to repository root"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Complete file content (required for create/update, omit for delete)"
                                },
                                "operation": {
                                    "type": "string",
                                    "enum": ["create", "update", "delete"],
                                    "description": "Operation: 'create' (new file), 'update' (modify existing), 'delete' (remove file)"
                                }
                            },
                            "required": ["path", "operation"]
                        },
                        "minItems": 1
                    },
                    "auto_generate_message": {
                        "type": "boolean",
                        "description": "If true, automatically generate conventional commit message(s) using AI. Default: false",
                        "default": False
                    },
                    "multi_commit": {
                        "type": "boolean",
                        "description": "If true, group changes into multiple logical commits. Recommended for >5 files. Default: false",
                        "default": False
                    },
                    "max_commits": {
                        "type": "integer",
                        "description": "Maximum commits to create in multi-commit mode. Default: 5",
                        "default": 5
                    }
                },
                "required": ["branch", "files"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Commit multiple file changes (single or multi-commit mode)"""
        github = _get_github_client()
        await github.connect()

        try:
            branch = input_data["branch"]
            commit_message = input_data.get("commit_message", "")
            files = input_data["files"]
            auto_generate = input_data.get("auto_generate_message", False)
            multi_commit = input_data.get("multi_commit", False)
            max_commits = input_data.get("max_commits", 5)

            # Validate at least one file
            if not files:
                return self._error_response(Exception("At least one file change is required"))

            # Multi-commit mode: Group changes into logical commits
            if multi_commit:
                return await self._multi_commit_flow(
                    github, branch, files, auto_generate, max_commits, context
                )

            # Single-commit mode (existing logic)
            # Generate commit message if requested
            if auto_generate and (not commit_message or commit_message.strip() == ""):
                # Check if we have LLM provider in context
                llm_provider = getattr(context, 'llm_provider', None)
                if not llm_provider:
                    return self._error_response(
                        Exception(
                            "AI commit message generation requires LLM provider. "
                            "Cannot auto-generate without LLM access."
                        )
                    )

                try:
                    # Create file changes context
                    file_changes = []
                    for file_info in files:
                        path = file_info["path"]
                        operation = file_info["operation"]
                        content = file_info.get("content", "")

                        file_change = FileChange(
                            path=path,
                            change_type=operation,
                            additions=len(content.split("\n")) if content else 0,
                            deletions=0,
                            diff_snippet=None
                        )
                        file_changes.append(file_change)

                    commit_context = CommitContext(
                        file_changes=file_changes,
                        branch_name=branch
                    )

                    # Generate message
                    logger.info(f"Generating commit message for {len(files)} file(s)")
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

            # Get branch HEAD
            try:
                branch_sha = await github.get_branch_sha(branch)
            except Exception as e:
                return self._error_response(
                    Exception(f"Branch '{branch}' not found. Create it first using create_branch tool.")
                )

            # Get current tree
            tree_data = await github.get_git_tree(branch_sha, recursive=False)
            base_tree_sha = tree_data["sha"]

            # Build tree changes
            tree_changes = []
            file_summaries = []

            for file_change in files:
                path = file_change["path"]
                operation = file_change["operation"]
                content = file_change.get("content")

                if operation in ["create", "update"]:
                    if content is None:
                        return self._error_response(
                            Exception(f"File '{path}': content is required for {operation} operation")
                        )

                    # Create blob for file content
                    blob_sha = await github.create_blob(content)

                    tree_changes.append({
                        "path": path,
                        "mode": "100644",  # Regular file
                        "type": "blob",
                        "sha": blob_sha
                    })

                    file_summaries.append(f"  - {operation.upper()}: {path} ({len(content)} bytes)")

                elif operation == "delete":
                    # Delete: set sha to null
                    tree_changes.append({
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": None  # Null SHA means delete
                    })

                    file_summaries.append(f"  - DELETE: {path}")

                else:
                    return self._error_response(
                        Exception(f"Invalid operation '{operation}' for file '{path}'")
                    )

            # Create tree with all changes
            new_tree_sha = await github.create_tree(base_tree_sha, tree_changes)

            # Create commit
            new_commit_sha = await github.create_commit(
                tree_sha=new_tree_sha,
                parent_sha=branch_sha,
                message=commit_message
            )

            # Update branch ref
            await github.update_branch_ref(branch, new_commit_sha)

            # Build success message
            result = f"Committed {len(files)} file(s) to branch '{branch}'\n"
            result += f"Commit: {new_commit_sha[:8]}\n"
            result += f"Message: {commit_message}\n\n"
            result += "Files:\n"
            result += "\n".join(file_summaries)
            result += "\n\n💡 **Next step**: Run `run_validation` to check for errors before creating a pull request."

            return self._success_response(
                result,
                metadata={
                    "branch": branch,
                    "commit_sha": new_commit_sha,
                    "file_count": len(files),
                    "files": [f["path"] for f in files]
                }
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()

    async def _multi_commit_flow(
        self,
        github: GitHubClient,
        branch: str,
        files: List[Dict[str, Any]],
        auto_generate: bool,
        max_commits: int,
        context: Any
    ) -> ToolResponse:
        """
        Execute multi-commit flow with intelligent grouping.

        Groups file changes into logical commits based on type, dependencies, and size.
        Falls back to single commit when grouping adds no value.

        Args:
            github: GitHub client instance
            branch: Target branch name
            files: List of file changes (path, content, operation)
            auto_generate: Whether to auto-generate commit messages
            max_commits: Maximum number of commits to create
            context: Agent context (for LLM provider access)

        Returns:
            ToolResponse with commit details or error
        """
        try:
            # 1. Convert input files to FileChange objects
            file_changes = []
            for file_info in files:
                path = file_info["path"]
                operation = file_info["operation"]
                content = file_info.get("content", "")

                file_change = FileChange(
                    path=path,
                    change_type=operation,
                    additions=len(content.split("\n")) if content else 0,
                    deletions=0,
                    diff_snippet=None
                )
                file_changes.append(file_change)

            # 2. Check if multi-commit is beneficial
            if not should_use_multi_commit(file_changes, min_files=5):
                logger.info("Multi-commit adds no value, falling back to single commit")
                # Fall back to single commit by calling parent logic
                single_files = files
                single_message = ""
                if auto_generate:
                    # Generate single commit message
                    llm_provider = getattr(context, 'llm_provider', None)
                    if llm_provider:
                        commit_context = CommitContext(
                            file_changes=file_changes,
                            branch_name=branch
                        )
                        generated = await generate_commit_message(
                            context=commit_context,
                            llm_provider=llm_provider,
                            temperature=0.3
                        )
                        single_message = generated.message

                # Re-route to single commit logic (lines 676-819)
                # This is intentional fallback
                input_data = {
                    "branch": branch,
                    "files": single_files,
                    "commit_message": single_message,
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

            # 4. Group files using strategies
            logger.info(f"Grouping {len(file_changes)} files into logical commits (max: {max_commits})")
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

            # 5. If only 1 group resulted, fall back to single commit
            if len(commit_groups) == 1:
                logger.info("Grouping resulted in 1 group, using single commit")
                input_data = {
                    "branch": branch,
                    "files": files,
                    "auto_generate_message": True
                }
                return await self.execute(input_data, context)

            # 6. Get current branch SHA
            try:
                current_sha = await github.get_branch_sha(branch)
            except Exception as e:
                return self._error_response(
                    Exception(f"Branch '{branch}' not found. Create it first using create_branch tool.")
                )

            # 7. Create commits sequentially
            commit_shas = []
            commit_summaries = []

            for group_index, group in enumerate(commit_groups):
                logger.info(f"Creating commit {group_index + 1}/{len(commit_groups)}: {group.commit_type.value} ({group.file_count} files)")

                # 7a. Generate commit message for this group
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

                # 7b. Get current tree
                tree_data = await github.get_git_tree(current_sha, recursive=False)
                base_tree_sha = tree_data["sha"]

                # 7c. Build tree changes for this group
                tree_changes = []
                for file_change in group.files:
                    # Find original file info from input
                    original_file = next((f for f in files if f["path"] == file_change.path), None)
                    if not original_file:
                        logger.warning(f"File {file_change.path} not found in original input")
                        continue

                    path = original_file["path"]
                    operation = original_file["operation"]
                    content = original_file.get("content")

                    if operation in ["create", "update"]:
                        if content is None:
                            logger.error(f"File '{path}': content is required for {operation} operation")
                            continue

                        # Create blob
                        blob_sha = await github.create_blob(content)

                        tree_changes.append({
                            "path": path,
                            "mode": "100644",
                            "type": "blob",
                            "sha": blob_sha
                        })

                    elif operation == "delete":
                        tree_changes.append({
                            "path": path,
                            "mode": "100644",
                            "type": "blob",
                            "sha": None  # Null SHA means delete
                        })

                # 7d. Create tree and commit
                new_tree_sha = await github.create_tree(base_tree_sha, tree_changes)
                new_commit_sha = await github.create_commit(
                    tree_sha=new_tree_sha,
                    parent_sha=current_sha,
                    message=message
                )

                # 7e. Update branch ref to point to new commit
                await github.update_branch_ref(branch, new_commit_sha)

                # 7f. Update current SHA for next iteration
                current_sha = new_commit_sha
                commit_shas.append(new_commit_sha)

                # 7g. Build summary for this commit
                commit_summaries.append({
                    "sha": new_commit_sha[:8],
                    "message": message.split("\n")[0],  # First line only
                    "type": group.commit_type.value,
                    "files": len(group.files),
                    "file_paths": [f.path for f in group.files],
                    "loc": group.total_loc
                })

                logger.info(f"Created commit {new_commit_sha[:8]}: {message.split(chr(10))[0]}")

            # 8. Build success message
            result = f"✅ Created {len(commit_shas)} commits with intelligent grouping on branch '{branch}'\\n\\n"
            result += "**Commits Created:**\\n"

            for i, summary in enumerate(commit_summaries, 1):
                result += f"\\n{i}. **{summary['sha']}** ({summary['type']})\\n"
                result += f"   {summary['message']}\\n"
                result += f"   Files: {summary['files']}, LOC: {summary['loc']}\\n"
                result += f"   Changed: {', '.join(summary['file_paths'][:3])}"
                if len(summary['file_paths']) > 3:
                    result += f" (+{len(summary['file_paths']) - 3} more)"
                result += "\\n"

            result += "\\n💡 **Next step**: Run `run_validation` to check for errors before creating a pull request."

            return self._success_response(
                result,
                metadata={
                    "branch": branch,
                    "commit_count": len(commit_shas),
                    "commit_shas": commit_shas,
                    "commits": commit_summaries,
                    "total_files": len(files),
                    "multi_commit": True
                }
            )

        except Exception as e:
            logger.error(f"Multi-commit operation failed: {e}")
            return self._error_response(
                Exception(f"Multi-commit operation failed: {e}")
            )
