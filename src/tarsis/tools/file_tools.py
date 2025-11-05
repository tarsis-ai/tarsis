"""
File operation tool handlers.

These tools allow the agent to work with files in the repository.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from .base import BaseToolHandler, ToolDefinition, ToolResponse, ToolCategory
from ..github import GitHubClient, GitHubConfig
from ..repository import RepositoryScanner, FileCategory, Language


# Global GitHub client instance
_github_client: Optional[GitHubClient] = None

# Global Repository scanner instance
_repo_scanner: Optional[RepositoryScanner] = None


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


def _get_repo_scanner() -> RepositoryScanner:
    """Get or create the repository scanner instance."""
    global _repo_scanner

    if _repo_scanner is None:
        github = _get_github_client()
        _repo_scanner = RepositoryScanner(github)

    return _repo_scanner


class ReadFileHandler(BaseToolHandler):
    """Tool to read file contents from the repository"""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Read the contents of a file from the repository. Use this to understand existing code before making changes.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file relative to repository root (e.g., 'src/main.py')"
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch to read from (defaults to default branch)",
                        "default": "main"
                    }
                },
                "required": ["file_path"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Read a file"""
        github = _get_github_client()
        await github.connect()

        try:
            file_path = input_data["file_path"]
            branch = input_data.get("branch") or await github.get_default_branch()

            # Read file from GitHub
            content = await github.get_file_content(file_path, ref=branch)

            if content is None:
                return self._success_response(
                    f"File not found: {file_path}",
                    metadata={"exists": False}
                )

            # Format response with file content
            result = f"""File: {file_path}
Branch: {branch}
Length: {len(content)} characters

---
{content}
---"""

            return self._success_response(result, metadata={"exists": True, "length": len(content)})

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()


class ListFilesHandler(BaseToolHandler):
    """Tool to list files in a directory"""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="List files in the repository. Can filter by directory path, file extension, or category. Useful for exploring repository structure.",
            input_schema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Filter by directory path (e.g., 'src', 'tests'). Leave empty for all files.",
                    },
                    "extension": {
                        "type": "string",
                        "description": "Filter by file extension (e.g., '.py', '.js')"
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "source_code", "test", "configuration", "documentation",
                            "build", "data", "asset", "script"
                        ],
                        "description": "Filter by file category"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to return (default: 50)",
                        "default": 50
                    }
                },
                "required": []
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: List files"""
        scanner = _get_repo_scanner()
        github = _get_github_client()
        await github.connect()

        try:
            # Get default branch
            default_branch = await github.get_default_branch()

            # Scan repository
            await scanner.scan_repository(ref=default_branch)

            # Get all files
            tree = await scanner.get_file_tree(default_branch)

            # Apply filters
            directory = input_data.get("directory", "").strip("/")
            extension = input_data.get("extension")
            category_str = input_data.get("category")
            limit = input_data.get("limit", 50)

            filtered_files = []
            for node in tree:
                # Only files, not directories
                if node.type != "blob":
                    continue

                # Directory filter
                if directory:
                    if not node.path.startswith(directory + "/") and node.path != directory:
                        continue

                # Extension filter
                if extension:
                    if not extension.startswith('.'):
                        extension = '.' + extension
                    if node.extension != extension.lower():
                        continue

                # Category filter
                if category_str:
                    try:
                        cat = FileCategory(category_str)
                        if node.category != cat:
                            continue
                    except ValueError:
                        pass

                filtered_files.append(node)

            # Limit results
            filtered_files = filtered_files[:limit]

            # Format response
            if not filtered_files:
                return self._success_response(
                    "No files found matching the criteria.",
                    metadata={"count": 0}
                )

            # Group by directory for better readability
            result_lines = [f"Found {len(filtered_files)} files:\n"]

            for node in filtered_files:
                size_str = f"{node.size} bytes" if node.size else "unknown size"
                cat_str = node.category.value if node.category else "unknown"
                result_lines.append(
                    f"  - {node.path} ({size_str}, {cat_str})"
                )

            result = "\n".join(result_lines)

            return self._success_response(
                result,
                metadata={
                    "count": len(filtered_files),
                    "truncated": len(tree) > limit
                }
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()


class SearchFilesHandler(BaseToolHandler):
    """Tool to search for files by name or content"""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE_ANALYSIS

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Search for files by name pattern using glob syntax (e.g., '**/*.py', 'src/**/test_*.py'). Great for finding specific files.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to search for (e.g., '**/*.py', 'src/**/*.js', '**/test_*')"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search should be case-sensitive (default: false)",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 30)",
                        "default": 30
                    }
                },
                "required": ["pattern"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Search files"""
        scanner = _get_repo_scanner()
        github = _get_github_client()
        await github.connect()

        try:
            pattern = input_data["pattern"]
            case_sensitive = input_data.get("case_sensitive", False)
            limit = input_data.get("limit", 30)

            # Get default branch
            default_branch = await github.get_default_branch()

            # Search files
            matches = await scanner.search_files(
                pattern=pattern,
                ref=default_branch,
                case_sensitive=case_sensitive
            )

            # Limit results
            matches = matches[:limit]

            if not matches:
                return self._success_response(
                    f"No files found matching pattern: {pattern}",
                    metadata={"count": 0, "pattern": pattern}
                )

            # Format response
            result_lines = [f"Found {len(matches)} files matching '{pattern}':\n"]

            for node in matches:
                lang_str = f", {node.language.value}" if node.language != Language.UNKNOWN else ""
                result_lines.append(
                    f"  - {node.path} ({node.category.value}{lang_str})"
                )

            result = "\n".join(result_lines)

            return self._success_response(
                result,
                metadata={
                    "count": len(matches),
                    "pattern": pattern,
                    "case_sensitive": case_sensitive
                }
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()

class GetRepositoryOverviewHandler(BaseToolHandler):
    """Tool to get a comprehensive repository overview"""

    @property
    def name(self) -> str:
        return "get_repository_overview"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE_ANALYSIS

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Get a comprehensive overview of the repository structure. Use this FIRST when working on a new repository to understand its organization, main directories, languages, and key files. This helps plan what files to read.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Get repository overview"""
        scanner = _get_repo_scanner()
        github = _get_github_client()
        await github.connect()

        try:
            # Get default branch
            default_branch = await github.get_default_branch()

            # Generate overview
            overview = await scanner.generate_overview(ref=default_branch)

            # Get cache stats
            cache_stats = scanner.get_cache_stats()

            return self._success_response(
                overview,
                metadata={
                    "branch": default_branch,
                    "cache_stats": cache_stats
                }
            )

        except Exception as e:
            return self._error_response(e)
        finally:
            await github.close()
