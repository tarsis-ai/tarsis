"""
Code search functionality using ripgrep.

Provides fast code search capabilities including:
- Content search within files
- Symbol finding (functions, classes, variables)
- Advanced regex pattern matching
- Result ranking and relevance scoring
"""

import os
import re
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from .file_types import FileTypeDetector, FileCategory, Language


class SearchType(Enum):
    """Type of search to perform"""
    TEXT = "text"  # Plain text search
    REGEX = "regex"  # Regular expression search
    SYMBOL = "symbol"  # Symbol (function/class) search
    EXACT = "exact"  # Exact match only


class SymbolType(Enum):
    """Type of code symbol"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    INTERFACE = "interface"
    TYPE = "type"


@dataclass
class SearchResult:
    """Represents a single search result"""
    file_path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    relevance_score: float = 0.0

    # Additional metadata
    language: Language = Language.UNKNOWN
    category: FileCategory = FileCategory.UNKNOWN
    symbol_type: Optional[SymbolType] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_start": self.match_start,
            "match_end": self.match_end,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "relevance_score": self.relevance_score,
            "language": self.language.value,
            "category": self.category.value,
            "symbol_type": self.symbol_type.value if self.symbol_type else None
        }


@dataclass
class SearchOptions:
    """Options for code search"""
    query: str
    search_type: SearchType = SearchType.TEXT
    case_sensitive: bool = False
    whole_word: bool = False
    file_pattern: Optional[str] = None  # e.g., "*.py"
    exclude_pattern: Optional[str] = None
    context_lines: int = 2
    max_results: int = 50
    sort_by: str = "relevance"  # relevance, file_path, line_number


class CodeSearcher:
    """
    Fast code search using ripgrep.

    Provides methods for searching code with various options and
    returns ranked results.
    """

    def __init__(self, repository_path: str):
        """
        Initialize code searcher.

        Args:
            repository_path: Path to local repository
        """
        self.repository_path = Path(repository_path)
        self._verify_ripgrep()

    def _verify_ripgrep(self) -> None:
        """Verify that ripgrep is installed"""
        try:
            subprocess.run(
                ["rg", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ripgrep (rg) is not installed or not in PATH. "
                "Install from: https://github.com/BurntSushi/ripgrep"
            )

    def search(self, options: SearchOptions) -> List[SearchResult]:
        """
        Perform code search with given options.

        Args:
            options: Search options

        Returns:
            List of search results, ranked by relevance
        """
        # Build ripgrep command
        cmd = self._build_rg_command(options)

        # Execute search
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=30
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Search timed out after 30 seconds")

        # Parse results
        results = self._parse_rg_output(result.stdout, options)

        # Rank results
        results = ResultRanker.rank_results(results, options.query)

        # Sort based on preference
        if options.sort_by == "relevance":
            results.sort(key=lambda r: r.relevance_score, reverse=True)
        elif options.sort_by == "file_path":
            results.sort(key=lambda r: (r.file_path, r.line_number))
        elif options.sort_by == "line_number":
            results.sort(key=lambda r: (r.line_number, r.file_path))

        # Limit results
        return results[:options.max_results]

    def _build_rg_command(self, options: SearchOptions) -> List[str]:
        """Build ripgrep command from options"""
        cmd = ["rg", "--json"]

        # Case sensitivity
        if not options.case_sensitive:
            cmd.append("--ignore-case")

        # Whole word matching
        if options.whole_word:
            cmd.append("--word-regexp")

        # Context lines
        if options.context_lines > 0:
            cmd.extend(["-C", str(options.context_lines)])

        # File pattern filtering
        if options.file_pattern:
            cmd.extend(["--glob", options.file_pattern])

        # Exclude pattern
        if options.exclude_pattern:
            cmd.extend(["--glob", f"!{options.exclude_pattern}"])

        # Default excludes
        cmd.extend([
            "--glob", "!.git",
            "--glob", "!node_modules",
            "--glob", "!__pycache__",
            "--glob", "!*.pyc",
            "--glob", "!.venv",
            "--glob", "!venv",
            "--glob", "!dist",
            "--glob", "!build"
        ])

        # Add the search pattern
        if options.search_type == SearchType.REGEX:
            cmd.append(options.query)
        else:
            # For text search, escape special regex characters
            escaped = re.escape(options.query)
            cmd.append(escaped)

        return cmd

    def _parse_rg_output(
        self,
        output: str,
        options: SearchOptions
    ) -> List[SearchResult]:
        """Parse ripgrep JSON output into SearchResult objects"""
        results = []
        current_match = None
        context_before = []
        context_after = []

        for line in output.strip().split('\n'):
            if not line:
                continue

            try:
                data = json.loads(line)
                msg_type = data.get("type")

                if msg_type == "match":
                    match_data = data.get("data", {})
                    path = match_data.get("path", {}).get("text", "")
                    line_num = match_data.get("line_number", 0)
                    line_text = match_data.get("lines", {}).get("text", "").rstrip('\n')

                    # Get match positions
                    submatches = match_data.get("submatches", [])
                    if submatches:
                        first_match = submatches[0]
                        match_start = first_match.get("start", 0)
                        match_end = first_match.get("end", 0)
                    else:
                        match_start = 0
                        match_end = len(line_text)

                    # Detect file type
                    language = FileTypeDetector.detect_language(path)
                    category = FileTypeDetector.detect_category(path)

                    result = SearchResult(
                        file_path=path,
                        line_number=line_num,
                        line_content=line_text,
                        match_start=match_start,
                        match_end=match_end,
                        context_before=context_before.copy(),
                        context_after=[],  # Will be populated by context messages
                        language=language,
                        category=category
                    )

                    results.append(result)
                    current_match = result
                    context_before = []

                elif msg_type == "context":
                    # Context lines (before or after match)
                    context_data = data.get("data", {})
                    line_text = context_data.get("lines", {}).get("text", "").rstrip('\n')

                    if current_match and len(current_match.context_after) < options.context_lines:
                        current_match.context_after.append(line_text)
                    else:
                        # Before context for next match
                        context_before.append(line_text)
                        if len(context_before) > options.context_lines:
                            context_before.pop(0)

            except json.JSONDecodeError:
                continue

        return results

    def search_text(
        self,
        query: str,
        file_pattern: Optional[str] = None,
        max_results: int = 50
    ) -> List[SearchResult]:
        """
        Simple text search.

        Args:
            query: Text to search for
            file_pattern: Optional file pattern filter
            max_results: Maximum results to return

        Returns:
            List of search results
        """
        options = SearchOptions(
            query=query,
            search_type=SearchType.TEXT,
            file_pattern=file_pattern,
            max_results=max_results
        )
        return self.search(options)

    def search_regex(
        self,
        pattern: str,
        file_pattern: Optional[str] = None,
        max_results: int = 50
    ) -> List[SearchResult]:
        """
        Regex pattern search.

        Args:
            pattern: Regex pattern
            file_pattern: Optional file pattern filter
            max_results: Maximum results to return

        Returns:
            List of search results
        """
        options = SearchOptions(
            query=pattern,
            search_type=SearchType.REGEX,
            file_pattern=file_pattern,
            max_results=max_results
        )
        return self.search(options)


class ResultRanker:
    """
    Ranks search results by relevance.

    Considers multiple factors:
    - Match quality (exact vs partial)
    - Match location (definition vs usage)
    - File importance (source > test > docs)
    - Match count per file
    """

    @staticmethod
    def rank_results(results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Rank search results by relevance.

        Args:
            results: List of search results
            query: Original search query

        Returns:
            Same list with relevance_score populated
        """
        query_lower = query.lower()

        for result in results:
            score = 0.0

            # Base score
            score += 1.0

            # 1. Match quality
            line_lower = result.line_content.lower()
            if query_lower in line_lower:
                # Exact match bonus
                if query in result.line_content:
                    score += 3.0
                else:
                    score += 2.0

                # Whole word match bonus
                if re.search(rf'\b{re.escape(query)}\b', result.line_content):
                    score += 2.0

            # 2. Match location (beginning of line = more important)
            if result.match_start < 10:
                score += 1.5

            # 3. File category importance
            if result.category == FileCategory.SOURCE_CODE:
                score += 2.0
            elif result.category == FileCategory.TEST:
                score += 1.0
            elif result.category == FileCategory.DOCUMENTATION:
                score += 0.5

            # 4. Detect if it's a definition (function/class)
            if ResultRanker._is_definition_line(result.line_content, result.language):
                score += 3.0

            # 5. File path depth (shorter = more important)
            depth = result.file_path.count('/')
            score += max(0, 5 - depth) * 0.5

            # 6. Penalize very long lines (likely minified or generated)
            if len(result.line_content) > 200:
                score -= 1.0

            result.relevance_score = score

        return results

    @staticmethod
    def _is_definition_line(line: str, language: Language) -> bool:
        """Check if line is likely a function/class definition"""
        line_stripped = line.strip()

        # Python
        if language == Language.PYTHON:
            if line_stripped.startswith(('def ', 'class ', 'async def ')):
                return True

        # JavaScript/TypeScript
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            if any(keyword in line_stripped for keyword in
                   ['function ', 'class ', 'const ', 'let ', 'var ']):
                if '=' in line_stripped or 'function' in line_stripped:
                    return True

        # Go
        elif language == Language.GO:
            if line_stripped.startswith('func '):
                return True

        # Java/C#
        elif language in (Language.JAVA, Language.CSHARP):
            if any(keyword in line_stripped for keyword in
                   ['class ', 'interface ', 'public ', 'private ', 'protected ']):
                return True

        # Rust
        elif language == Language.RUST:
            if line_stripped.startswith(('fn ', 'struct ', 'enum ', 'trait ', 'impl ')):
                return True

        return False


# Language-specific symbol patterns
SYMBOL_PATTERNS = {
    Language.PYTHON: {
        SymbolType.FUNCTION: r'^\s*def\s+{symbol}\s*\(',
        SymbolType.CLASS: r'^\s*class\s+{symbol}\s*[:\(]',
        SymbolType.IMPORT: r'^\s*(?:from\s+\S+\s+)?import\s+.*\b{symbol}\b',
    },
    Language.JAVASCRIPT: {
        SymbolType.FUNCTION: r'function\s+{symbol}\s*\(',
        SymbolType.CLASS: r'class\s+{symbol}\s*[{{]',
        SymbolType.CONSTANT: r'const\s+{symbol}\s*=',
        SymbolType.VARIABLE: r'(?:let|var)\s+{symbol}\s*=',
    },
    Language.TYPESCRIPT: {
        SymbolType.FUNCTION: r'function\s+{symbol}\s*[<(]',
        SymbolType.CLASS: r'class\s+{symbol}\s*[<{{]',
        SymbolType.INTERFACE: r'interface\s+{symbol}\s*[{{]',
        SymbolType.TYPE: r'type\s+{symbol}\s*=',
    },
    Language.GO: {
        SymbolType.FUNCTION: r'func\s+(?:\([^)]*\)\s+)?{symbol}\s*\(',
        SymbolType.TYPE: r'type\s+{symbol}\s+(?:struct|interface)',
    },
    Language.JAVA: {
        SymbolType.CLASS: r'class\s+{symbol}\s*[{{<]',
        SymbolType.INTERFACE: r'interface\s+{symbol}\s*[{{<]',
        SymbolType.METHOD: r'(?:public|private|protected).*\s+{symbol}\s*\(',
    },
    Language.RUST: {
        SymbolType.FUNCTION: r'fn\s+{symbol}\s*[<(]',
        SymbolType.TYPE: r'(?:struct|enum|trait)\s+{symbol}\s*[{{<]',
    },
}


class SymbolFinder:
    """
    Finds code symbols (functions, classes, etc.) in repositories.
    """

    def __init__(self, searcher: CodeSearcher):
        """
        Initialize symbol finder.

        Args:
            searcher: CodeSearcher instance
        """
        self.searcher = searcher

    def find_symbol(
        self,
        symbol_name: str,
        symbol_type: Optional[SymbolType] = None,
        language: Optional[Language] = None,
        exact_match: bool = True
    ) -> List[SearchResult]:
        """
        Find symbol definitions in code.

        Args:
            symbol_name: Name of the symbol to find
            symbol_type: Type of symbol (function, class, etc.)
            language: Programming language to search in
            exact_match: Whether to match exact symbol name

        Returns:
            List of search results with symbol definitions
        """
        results = []

        # Determine which languages to search
        languages = [language] if language else list(SYMBOL_PATTERNS.keys())

        for lang in languages:
            if lang not in SYMBOL_PATTERNS:
                continue

            patterns = SYMBOL_PATTERNS[lang]

            # Determine which symbol types to search for
            symbol_types = [symbol_type] if symbol_type else list(patterns.keys())

            for sym_type in symbol_types:
                if sym_type not in patterns:
                    continue

                # Build pattern
                pattern_template = patterns[sym_type]

                if exact_match:
                    pattern = pattern_template.format(symbol=re.escape(symbol_name))
                else:
                    pattern = pattern_template.format(symbol=f'\\w*{re.escape(symbol_name)}\\w*')

                # Get file extension for language
                file_pattern = self._get_file_pattern_for_language(lang)

                # Search
                search_results = self.searcher.search_regex(
                    pattern=pattern,
                    file_pattern=file_pattern,
                    max_results=100
                )

                # Mark symbol type
                for result in search_results:
                    result.symbol_type = sym_type

                results.extend(search_results)

        # Remove duplicates (same file and line)
        seen = set()
        unique_results = []
        for result in results:
            key = (result.file_path, result.line_number)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        # Rank by relevance
        unique_results = ResultRanker.rank_results(unique_results, symbol_name)
        unique_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return unique_results

    def _get_file_pattern_for_language(self, language: Language) -> str:
        """Get file glob pattern for a language"""
        patterns = {
            Language.PYTHON: "*.py",
            Language.JAVASCRIPT: "*.js",
            Language.TYPESCRIPT: "*.{ts,tsx}",
            Language.GO: "*.go",
            Language.JAVA: "*.java",
            Language.RUST: "*.rs",
            Language.CSHARP: "*.cs",
            Language.CPP: "*.{cpp,cc,cxx,hpp,h}",
            Language.C: "*.{c,h}",
            Language.RUBY: "*.rb",
            Language.PHP: "*.php",
        }
        return patterns.get(language, "*")

    def find_function(self, function_name: str, language: Optional[Language] = None) -> List[SearchResult]:
        """Find function definitions"""
        return self.find_symbol(function_name, SymbolType.FUNCTION, language)

    def find_class(self, class_name: str, language: Optional[Language] = None) -> List[SearchResult]:
        """Find class definitions"""
        return self.find_symbol(class_name, SymbolType.CLASS, language)

    def find_imports(self, import_name: str, language: Optional[Language] = None) -> List[SearchResult]:
        """Find import statements"""
        return self.find_symbol(import_name, SymbolType.IMPORT, language)
