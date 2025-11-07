"""
Repository scanning and analysis module.

Provides tools for discovering and analyzing repository structure,
enabling the agent to navigate codebases efficiently.
"""

from .scanner import RepositoryScanner, FileTreeNode
from .cache import RepositoryCache
from .file_types import FileTypeDetector, FileCategory, Language
from .search import (
    CodeSearcher,
    SymbolFinder,
    ResultRanker,
    SearchResult,
    SearchOptions,
    SearchType,
    SymbolType
)
from .discovery import (
    HybridDiscoveryEngine,
    FileDiscoveryResult,
    DiscoveryStrategy
)
from .local import CloneManager, CloneError
from .file_operations import (
    rename_file,
    create_symlink,
    can_create_symlinks,
    safe_push,
    batch_modify_files,
    FileOperationError
)

__all__ = [
    # Scanner
    "RepositoryScanner",
    "FileTreeNode",
    "RepositoryCache",
    "FileTypeDetector",
    "FileCategory",
    "Language",
    # Search
    "CodeSearcher",
    "SymbolFinder",
    "ResultRanker",
    "SearchResult",
    "SearchOptions",
    "SearchType",
    "SymbolType",
    # Discovery
    "HybridDiscoveryEngine",
    "FileDiscoveryResult",
    "DiscoveryStrategy",
    # Local clone management
    "CloneManager",
    "CloneError",
    # File operations
    "rename_file",
    "create_symlink",
    "can_create_symlinks",
    "safe_push",
    "batch_modify_files",
    "FileOperationError"
]
