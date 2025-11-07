"""
Code search tool handlers.

These tools allow the agent to search within file contents,
find symbols (functions, classes), and perform advanced pattern matching.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

from .base import BaseToolHandler, ToolDefinition, ToolResponse, ToolCategory
from ..repository import (
    CodeSearcher,
    SymbolFinder,
    SearchOptions,
    SearchType,
    SymbolType,
    Language
)
from ..github import GitHubClient, GitHubConfig
import subprocess


# Global instances (DEPRECATED - kept for backward compatibility)
_code_searcher: Optional[CodeSearcher] = None
_symbol_finder: Optional[SymbolFinder] = None
_repo_clone_path: Optional[str] = None


def _get_or_clone_repository() -> str:
    """
    DEPRECATED: Get or clone the repository locally for searching.

    This function is kept for backward compatibility but should not be used.
    Use CloneManager from context instead.

    Returns:
        Path to local repository
    """
    global _repo_clone_path

    if _repo_clone_path and Path(_repo_clone_path).exists():
        return _repo_clone_path

    # Create temporary directory for repo
    temp_dir = tempfile.mkdtemp(prefix="tarsis_repo_")

    # Get repo info from environment
    owner = os.getenv("GITHUB_REPO_OWNER", "")
    name = os.getenv("GITHUB_REPO_NAME", "")
    token = os.getenv("GITHUB_TOKEN", "")

    if not owner or not name:
        raise ValueError("GITHUB_REPO_OWNER and GITHUB_REPO_NAME must be set")

    # Clone repository
    repo_url = f"https://github.com/{owner}/{name}.git"
    if token:
        # Use token for authentication
        repo_url = f"https://{token}@github.com/{owner}/{name}.git"

    clone_path = os.path.join(temp_dir, name)

    try:
        # Shallow clone for speed (depth=1)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, clone_path],
            check=True,
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
        _repo_clone_path = clone_path
        return clone_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repository: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Repository clone timed out after 5 minutes")


def _get_code_searcher() -> CodeSearcher:
    """
    DEPRECATED: Get or create the code searcher instance.

    Kept for backward compatibility.
    """
    global _code_searcher

    if _code_searcher is None:
        repo_path = _get_or_clone_repository()
        _code_searcher = CodeSearcher(repo_path)

    return _code_searcher


def _get_symbol_finder() -> SymbolFinder:
    """
    DEPRECATED: Get or create the symbol finder instance.

    Kept for backward compatibility.
    """
    global _symbol_finder

    if _symbol_finder is None:
        searcher = _get_code_searcher()
        _symbol_finder = SymbolFinder(searcher)

    return _symbol_finder


async def _get_repo_path_from_context(context: Any) -> str:
    """
    Get repository path from context using clone manager.

    Args:
        context: Task context with clone_manager

    Returns:
        Repository path as string

    Raises:
        RuntimeError: If unable to get repository path
    """
    # Try to use clone manager
    if hasattr(context, "clone_manager") and context.clone_manager:
        try:
            # Ensure clone exists
            repo_path = await context.clone_manager.ensure_clone(shallow=True)
            return repo_path
        except Exception as e:
            # Clone failed - fall through to deprecated method
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to use clone manager: {e}, falling back to legacy method")

    # Fallback to deprecated global method (for backward compatibility)
    return _get_or_clone_repository()


class SearchCodeHandler(BaseToolHandler):
    """Tool to search within file contents"""

    @property
    def name(self) -> str:
        return "search_code"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE_ANALYSIS

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Search for text or patterns within file contents across the repository. "
                "This is much more powerful than search_files which only searches filenames. "
                "Use this to find specific code snippets, variable usages, function calls, "
                "error messages, or any text within the code."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or pattern to search for in file contents"
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Whether to treat query as a regex pattern (default: false)",
                        "default": False
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search should be case-sensitive (default: false)",
                        "default": False
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Filter by file pattern (e.g., '*.py', '*.{js,ts}'). Leave empty for all files."
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines to show before/after match (default: 2)",
                        "default": 2
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 30)",
                        "default": 30
                    }
                },
                "required": ["query"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Search code"""
        try:
            query = input_data["query"]
            is_regex = input_data.get("regex", False)
            case_sensitive = input_data.get("case_sensitive", False)
            file_pattern = input_data.get("file_pattern")
            context_lines = input_data.get("context_lines", 2)
            max_results = input_data.get("max_results", 30)

            # Get repository path (using clone manager if available)
            repo_path = await _get_repo_path_from_context(context)

            # Create searcher for this request
            searcher = CodeSearcher(repo_path)

            # Build search options
            options = SearchOptions(
                query=query,
                search_type=SearchType.REGEX if is_regex else SearchType.TEXT,
                case_sensitive=case_sensitive,
                file_pattern=file_pattern,
                context_lines=context_lines,
                max_results=max_results,
                sort_by="relevance"
            )

            # Execute search
            results = searcher.search(options)

            if not results:
                return self._success_response(
                    f"No matches found for: {query}",
                    metadata={"count": 0, "query": query}
                )

            # Format results
            result_lines = [
                f"Found {len(results)} matches for '{query}' (showing top {min(len(results), max_results)}):\n"
            ]

            for i, result in enumerate(results, 1):
                result_lines.append(f"\n{i}. {result.file_path}:{result.line_number}")
                result_lines.append(f"   Relevance: {result.relevance_score:.2f}")

                # Show context before
                if result.context_before:
                    for ctx_line in result.context_before[-2:]:  # Last 2 lines
                        result_lines.append(f"   | {ctx_line}")

                # Show matched line with highlight
                matched_line = result.line_content
                result_lines.append(f"   > {matched_line}")

                # Show context after
                if result.context_after:
                    for ctx_line in result.context_after[:2]:  # First 2 lines
                        result_lines.append(f"   | {ctx_line}")

            result_text = "\n".join(result_lines)

            return self._success_response(
                result_text,
                metadata={
                    "count": len(results),
                    "query": query,
                    "is_regex": is_regex,
                    "files": list(set(r.file_path for r in results))
                }
            )

        except Exception as e:
            return self._error_response(e)


class FindSymbolHandler(BaseToolHandler):
    """Tool to find symbol definitions (functions, classes, etc.)"""

    @property
    def name(self) -> str:
        return "find_symbol"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE_ANALYSIS

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Find definitions of code symbols like functions, classes, methods, or interfaces. "
                "This is useful when you need to understand where something is defined or "
                "see the implementation of a specific function or class. "
                "Supports multiple programming languages."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to find (e.g., 'login', 'UserClass', 'calculate_total')"
                    },
                    "symbol_type": {
                        "type": "string",
                        "enum": ["function", "class", "method", "interface", "type"],
                        "description": "Type of symbol to search for. Leave empty to search all types."
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "typescript", "go", "java", "rust", "csharp"],
                        "description": "Programming language to search in. Leave empty to search all languages."
                    },
                    "exact_match": {
                        "type": "boolean",
                        "description": "Whether to match exact symbol name only (default: true)",
                        "default": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["symbol_name"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Find symbol"""
        try:
            symbol_name = input_data["symbol_name"]
            symbol_type_str = input_data.get("symbol_type")
            language_str = input_data.get("language")
            exact_match = input_data.get("exact_match", True)
            max_results = input_data.get("max_results", 20)

            # Parse symbol type
            symbol_type = None
            if symbol_type_str:
                try:
                    symbol_type = SymbolType(symbol_type_str.lower())
                except ValueError:
                    return self._error_response(
                        ValueError(f"Invalid symbol_type: {symbol_type_str}")
                    )

            # Parse language
            language = None
            if language_str:
                try:
                    language = Language(language_str.upper())
                except ValueError:
                    return self._error_response(
                        ValueError(f"Invalid language: {language_str}")
                    )

            # Get repository path (using clone manager if available)
            repo_path = await _get_repo_path_from_context(context)

            # Create searcher and symbol finder for this request
            searcher = CodeSearcher(repo_path)
            finder = SymbolFinder(searcher)

            # Find symbols
            results = finder.find_symbol(
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                language=language,
                exact_match=exact_match
            )

            # Limit results
            results = results[:max_results]

            if not results:
                search_desc = f"'{symbol_name}'"
                if symbol_type:
                    search_desc += f" ({symbol_type.value})"
                if language:
                    search_desc += f" in {language.value}"

                return self._success_response(
                    f"No symbol definitions found for: {search_desc}",
                    metadata={"count": 0, "symbol_name": symbol_name}
                )

            # Format results
            result_lines = [
                f"Found {len(results)} definition(s) for '{symbol_name}':\n"
            ]

            for i, result in enumerate(results, 1):
                symbol_info = f"{result.symbol_type.value}" if result.symbol_type else "symbol"
                result_lines.append(
                    f"\n{i}. {symbol_info} in {result.file_path}:{result.line_number}"
                )
                result_lines.append(f"   Language: {result.language.value}")
                result_lines.append(f"   Relevance: {result.relevance_score:.2f}")
                result_lines.append(f"   Definition:")

                # Show context before
                if result.context_before:
                    for ctx_line in result.context_before[-1:]:  # Last line
                        result_lines.append(f"     {ctx_line}")

                # Show definition line
                result_lines.append(f"   > {result.line_content}")

                # Show context after
                if result.context_after:
                    for ctx_line in result.context_after[:3]:  # First 3 lines
                        result_lines.append(f"     {ctx_line}")

            result_text = "\n".join(result_lines)

            return self._success_response(
                result_text,
                metadata={
                    "count": len(results),
                    "symbol_name": symbol_name,
                    "symbol_type": symbol_type.value if symbol_type else None,
                    "files": list(set(r.file_path for r in results))
                }
            )

        except Exception as e:
            return self._error_response(e)


class GrepPatternHandler(BaseToolHandler):
    """Tool for advanced regex pattern search"""

    @property
    def name(self) -> str:
        return "grep_pattern"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE_ANALYSIS

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Advanced regex pattern search across the codebase. "
                "Use this for complex pattern matching when search_code is not sufficient. "
                "Supports full regex syntax and provides ranked results. "
                "Useful for finding patterns like 'all functions that call X', "
                "'all files importing Y', 'error handling patterns', etc."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (supports full regex syntax)"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Filter by file pattern (e.g., '*.py', '*.{js,ts}')"
                    },
                    "exclude_pattern": {
                        "type": "string",
                        "description": "Exclude files matching this pattern (e.g., '*test*', '*.min.js')"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines to show (default: 2)",
                        "default": 2
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 30)",
                        "default": 30
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "file_path", "line_number"],
                        "description": "How to sort results (default: relevance)",
                        "default": "relevance"
                    }
                },
                "required": ["pattern"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Grep pattern"""
        try:
            pattern = input_data["pattern"]
            file_pattern = input_data.get("file_pattern")
            exclude_pattern = input_data.get("exclude_pattern")
            context_lines = input_data.get("context_lines", 2)
            max_results = input_data.get("max_results", 30)
            sort_by = input_data.get("sort_by", "relevance")

            # Get repository path (using clone manager if available)
            repo_path = await _get_repo_path_from_context(context)

            # Create searcher for this request
            searcher = CodeSearcher(repo_path)

            # Build search options
            options = SearchOptions(
                query=pattern,
                search_type=SearchType.REGEX,
                case_sensitive=True,  # Regex is typically case-sensitive
                file_pattern=file_pattern,
                exclude_pattern=exclude_pattern,
                context_lines=context_lines,
                max_results=max_results,
                sort_by=sort_by
            )

            # Execute search
            results = searcher.search(options)

            if not results:
                return self._success_response(
                    f"No matches found for pattern: {pattern}",
                    metadata={"count": 0, "pattern": pattern}
                )

            # Format results
            result_lines = [
                f"Found {len(results)} matches for pattern '{pattern}':\n"
            ]

            for i, result in enumerate(results, 1):
                result_lines.append(f"\n{i}. {result.file_path}:{result.line_number}")
                if sort_by == "relevance":
                    result_lines.append(f"   Relevance: {result.relevance_score:.2f}")

                # Show context
                if result.context_before:
                    for ctx_line in result.context_before[-2:]:
                        result_lines.append(f"   | {ctx_line}")

                result_lines.append(f"   > {result.line_content}")

                if result.context_after:
                    for ctx_line in result.context_after[:2]:
                        result_lines.append(f"   | {ctx_line}")

            result_text = "\n".join(result_lines)

            # Get file distribution
            files = {}
            for r in results:
                files[r.file_path] = files.get(r.file_path, 0) + 1

            return self._success_response(
                result_text,
                metadata={
                    "count": len(results),
                    "pattern": pattern,
                    "files": files,
                    "sort_by": sort_by
                }
            )

        except Exception as e:
            return self._error_response(e)
