"""
Repository structure scanner.

Provides functionality to scan and analyze repository file trees
using GitHub's Git Trees API.
"""

import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import fnmatch

from ..github import GitHubClient, GitHubConfig
from .cache import RepositoryCache
from .file_types import FileTypeDetector, FileCategory, Language


@dataclass
class FileTreeNode:
    """Represents a file or directory in the repository tree"""
    path: str
    type: str  # "blob" (file) or "tree" (directory)
    size: Optional[int] = None
    sha: Optional[str] = None
    mode: Optional[str] = None

    # Derived properties
    name: str = field(init=False)
    extension: str = field(init=False)
    category: FileCategory = field(init=False)
    language: Language = field(init=False)
    is_binary: bool = field(init=False)

    def __post_init__(self):
        """Initialize derived properties"""
        self.name = os.path.basename(self.path)
        self.extension = os.path.splitext(self.path)[1].lower()
        self.category = FileTypeDetector.detect_category(self.path)
        self.language = FileTypeDetector.detect_language(self.path)
        self.is_binary = FileTypeDetector.is_binary(self.path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "path": self.path,
            "name": self.name,
            "type": self.type,
            "size": self.size,
            "extension": self.extension,
            "category": self.category.value if isinstance(self.category, FileCategory) else self.category,
            "language": self.language.value if isinstance(self.language, Language) else self.language,
            "is_binary": self.is_binary
        }


class RepositoryScanner:
    """
    Scans and analyzes repository structure.

    Uses GitHub's Git Trees API to efficiently fetch the entire repository
    tree and provides methods for querying and analyzing the structure.
    """

    def __init__(
        self,
        github_client: GitHubClient,
        cache: Optional[RepositoryCache] = None
    ):
        """
        Initialize scanner.

        Args:
            github_client: GitHub API client
            cache: Optional cache instance (creates new if None)
        """
        self.github = github_client
        self.cache = cache or RepositoryCache()
        self._tree: Optional[List[FileTreeNode]] = None
        self._tree_dict: Dict[str, FileTreeNode] = {}

    async def scan_repository(
        self,
        ref: str = "main",
        force_refresh: bool = False
    ) -> List[FileTreeNode]:
        """
        Scan repository and build file tree.

        Args:
            ref: Branch, tag, or commit ref to scan
            force_refresh: Force refresh even if cached

        Returns:
            List of file tree nodes
        """
        # Get commit SHA for the ref
        commit_sha = await self.github.get_branch_sha(ref)

        # Check cache
        if not force_refresh:
            cached_data = self.cache.get(
                self.github.config.repo_owner,
                self.github.config.repo_name,
                ref,
                commit_sha
            )
            if cached_data:
                self._tree = [
                    FileTreeNode(**node) if isinstance(node, dict) else node
                    for node in cached_data
                ]
                self._build_tree_dict()
                return self._tree

        # Fetch git tree from GitHub
        git_tree = await self.github.get_git_tree(commit_sha, recursive=True)

        # Build file tree nodes
        nodes = []
        for item in git_tree.get("tree", []):
            node = FileTreeNode(
                path=item["path"],
                type=item["type"],
                size=item.get("size"),
                sha=item.get("sha"),
                mode=item.get("mode")
            )

            # Skip excluded files
            if FileTypeDetector.should_exclude(node.path):
                continue

            nodes.append(node)

        self._tree = nodes
        self._build_tree_dict()

        # Cache the results
        cache_data = [node.to_dict() for node in nodes]
        self.cache.set(
            self.github.config.repo_owner,
            self.github.config.repo_name,
            ref,
            commit_sha,
            cache_data
        )

        return self._tree

    def _build_tree_dict(self):
        """Build dictionary for fast path lookup"""
        if self._tree:
            self._tree_dict = {node.path: node for node in self._tree}

    async def get_file_tree(self, ref: str = "main") -> List[FileTreeNode]:
        """
        Get repository file tree.

        Args:
            ref: Branch/tag to get tree for

        Returns:
            List of file tree nodes
        """
        if self._tree is None:
            await self.scan_repository(ref)
        return self._tree or []

    async def get_files_by_category(
        self,
        category: FileCategory,
        ref: str = "main"
    ) -> List[FileTreeNode]:
        """
        Get files filtered by category.

        Args:
            category: File category to filter by
            ref: Branch/tag

        Returns:
            List of matching files
        """
        tree = await self.get_file_tree(ref)
        return [
            node for node in tree
            if node.type == "blob" and node.category == category
        ]

    async def get_files_by_extension(
        self,
        extension: str,
        ref: str = "main"
    ) -> List[FileTreeNode]:
        """
        Get files filtered by extension.

        Args:
            extension: File extension (e.g., '.py', '.js')
            ref: Branch/tag

        Returns:
            List of matching files
        """
        if not extension.startswith('.'):
            extension = '.' + extension

        tree = await self.get_file_tree(ref)
        return [
            node for node in tree
            if node.type == "blob" and node.extension == extension.lower()
        ]

    async def get_files_by_language(
        self,
        language: Language,
        ref: str = "main"
    ) -> List[FileTreeNode]:
        """
        Get files filtered by programming language.

        Args:
            language: Programming language
            ref: Branch/tag

        Returns:
            List of matching files
        """
        tree = await self.get_file_tree(ref)
        return [
            node for node in tree
            if node.type == "blob" and node.language == language
        ]

    async def search_files(
        self,
        pattern: str,
        ref: str = "main",
        case_sensitive: bool = False
    ) -> List[FileTreeNode]:
        """
        Search for files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '**/*.py', 'src/**/*.js')
            ref: Branch/tag
            case_sensitive: Whether search should be case-sensitive

        Returns:
            List of matching files
        """
        tree = await self.get_file_tree(ref)

        if not case_sensitive:
            pattern = pattern.lower()

        matches = []
        for node in tree:
            if node.type != "blob":
                continue

            path_to_match = node.path if case_sensitive else node.path.lower()

            if fnmatch.fnmatch(path_to_match, pattern):
                matches.append(node)

        return matches

    async def get_directory_structure(
        self,
        max_depth: Optional[int] = None,
        ref: str = "main"
    ) -> Dict[str, Any]:
        """
        Get hierarchical directory structure.

        Args:
            max_depth: Maximum depth to traverse (None for unlimited)
            ref: Branch/tag

        Returns:
            Nested dictionary representing directory structure
        """
        tree = await self.get_file_tree(ref)

        root = {"type": "directory", "name": "/", "children": {}}

        for node in tree:
            parts = node.path.split("/")

            # Check depth limit
            if max_depth is not None and len(parts) > max_depth:
                continue

            # Navigate/create nested structure
            current = root
            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1

                if is_last:
                    # It's a file
                    current["children"][part] = {
                        "type": "file",
                        "name": part,
                        "path": node.path,
                        "size": node.size,
                        "category": node.category.value,
                        "language": node.language.value
                    }
                else:
                    # It's a directory
                    if part not in current["children"]:
                        current["children"][part] = {
                            "type": "directory",
                            "name": part,
                            "children": {}
                        }
                    current = current["children"][part]

        return root

    async def generate_overview(self, ref: str = "main") -> str:
        """
        Generate a comprehensive repository overview for LLM context.

        Args:
            ref: Branch/tag

        Returns:
            Formatted overview text
        """
        tree = await self.get_file_tree(ref)

        if not tree:
            return "Repository is empty or could not be scanned."

        # Collect statistics
        total_files = sum(1 for node in tree if node.type == "blob")
        total_dirs = sum(1 for node in tree if node.type == "tree")

        # Group by category
        by_category: Dict[FileCategory, List[FileTreeNode]] = defaultdict(list)
        for node in tree:
            if node.type == "blob":
                by_category[node.category].append(node)

        # Group by language
        by_language: Dict[Language, List[FileTreeNode]] = defaultdict(list)
        for node in tree:
            if node.type == "blob" and node.language != Language.UNKNOWN:
                by_language[node.language].append(node)

        # Top-level directories
        top_dirs = set()
        for node in tree:
            parts = node.path.split("/")
            if len(parts) > 1:
                top_dirs.add(parts[0])

        # Build overview
        lines = []
        lines.append("# Repository Overview")
        lines.append("")
        lines.append(f"**Repository**: {self.github.config.repo_owner}/{self.github.config.repo_name}")
        lines.append(f"**Branch**: {ref}")
        lines.append("")

        lines.append("## Statistics")
        lines.append(f"- Total files: {total_files}")
        lines.append(f"- Total directories: {total_dirs}")
        lines.append("")

        lines.append("## Top-level Directories")
        for dir_name in sorted(top_dirs):
            lines.append(f"- `{dir_name}/`")
        lines.append("")

        if by_language:
            lines.append("## Languages")
            for lang in sorted(by_language.keys(), key=lambda l: len(by_language[l]), reverse=True):
                count = len(by_language[lang])
                lines.append(f"- {lang.value}: {count} files")
            lines.append("")

        lines.append("## File Categories")
        for category in sorted(by_category.keys(), key=lambda c: len(by_category[c]), reverse=True):
            count = len(by_category[category])
            if count > 0:
                lines.append(f"- {category.value}: {count} files")
        lines.append("")

        # Key configuration files
        config_files = [
            node for node in tree
            if node.type == "blob" and node.category == FileCategory.CONFIGURATION
            and "/" not in node.path  # Root level only
        ]
        if config_files:
            lines.append("## Key Configuration Files")
            for node in sorted(config_files, key=lambda n: n.name):
                lines.append(f"- `{node.name}`")
            lines.append("")

        # Documentation files
        doc_files = [
            node for node in tree
            if node.type == "blob" and node.category == FileCategory.DOCUMENTATION
            and "/" not in node.path  # Root level only
        ]
        if doc_files:
            lines.append("## Documentation")
            for node in sorted(doc_files, key=lambda n: n.name):
                lines.append(f"- `{node.name}`")
            lines.append("")

        # Source code structure
        source_files = by_category.get(FileCategory.SOURCE_CODE, [])
        if source_files:
            # Group source files by top-level directory
            source_by_dir: Dict[str, int] = defaultdict(int)
            for node in source_files:
                parts = node.path.split("/")
                top_dir = parts[0] if len(parts) > 1 else "root"
                source_by_dir[top_dir] += 1

            if source_by_dir:
                lines.append("## Source Code Structure")
                for dir_name in sorted(source_by_dir.keys(), key=lambda d: source_by_dir[d], reverse=True):
                    count = source_by_dir[dir_name]
                    lines.append(f"- `{dir_name}/`: {count} source files")
                lines.append("")

        # Test files
        test_files = by_category.get(FileCategory.TEST, [])
        if test_files:
            lines.append(f"## Tests")
            lines.append(f"- {len(test_files)} test files found")
            lines.append("")

        return "\n".join(lines)

    def get_node(self, path: str) -> Optional[FileTreeNode]:
        """
        Get a specific file tree node by path.

        Args:
            path: File path

        Returns:
            File tree node or None
        """
        return self._tree_dict.get(path)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
