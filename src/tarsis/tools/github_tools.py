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
from typing import Dict, Any, Optional
from .base import BaseToolHandler, ToolDefinition, ToolResponse, ToolCategory
from ..github import GitHubClient, GitHubConfig


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

💡 **Reminder**: After modifying files, use `run_validation` to check for errors before creating a pull request.""",
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
                        "description": "Commit message describing the change"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["create", "update", "auto"],
                        "description": "Operation type: 'create' (file must not exist), 'update' (file must exist), 'auto' (detect automatically). Default: auto",
                        "default": "auto"
                    }
                },
                "required": ["file_path", "content", "branch", "commit_message"]
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
            commit_message = input_data["commit_message"]
            operation = input_data.get("operation", "auto")

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
            description="""Commit multiple file changes (create/update/delete) atomically in a single commit. Use this when implementing features that require changes to multiple files.

💡 **Reminder**: After committing changes, use `run_validation` to check for errors before creating a pull request.""",
            input_schema={
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch to commit to (must already exist)"
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message describing all changes"
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
                    }
                },
                "required": ["branch", "commit_message", "files"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Commit multiple file changes"""
        github = _get_github_client()
        await github.connect()

        try:
            branch = input_data["branch"]
            commit_message = input_data["commit_message"]
            files = input_data["files"]

            # Validate at least one file
            if not files:
                return self._error_response(Exception("At least one file change is required"))

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
