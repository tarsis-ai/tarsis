"""
Hybrid file discovery engine.

Combines multiple search strategies with LLM reasoning to intelligently
discover relevant files in a repository.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .scanner import RepositoryScanner
from .search import CodeSearcher, SymbolFinder, SearchResult
from .file_types import Language, FileCategory
from ..llm.provider import ILLMProvider


logger = logging.getLogger(__name__)


class DiscoveryStrategy(Enum):
    """Search strategies for file discovery"""
    FILENAME = "filename"  # Search filenames only
    CONTENT = "content"    # Search file contents
    SYMBOL = "symbol"      # Search symbols (functions/classes)
    COMBINED = "combined"  # Use all strategies


@dataclass
class FileDiscoveryResult:
    """Represents a discovered file with relevance information"""
    file_path: str
    relevance_score: float  # 0.0 - 1.0
    match_count: int
    snippet: str  # Most relevant snippet from file
    reasoning: str  # Why this file is relevant
    match_types: List[str] = field(default_factory=list)  # e.g., ["content", "symbol"]
    language: Language = Language.UNKNOWN
    category: FileCategory = FileCategory.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_path": self.file_path,
            "relevance_score": self.relevance_score,
            "match_count": self.match_count,
            "snippet": self.snippet,
            "reasoning": self.reasoning,
            "match_types": self.match_types,
            "language": self.language.value,
            "category": self.category.value
        }


@dataclass
class FileMatchData:
    """Aggregated match data for a single file"""
    file_path: str
    filename_matched: bool = False
    content_matches: List[SearchResult] = field(default_factory=list)
    symbol_matches: List[SearchResult] = field(default_factory=list)
    total_matches: int = 0
    language: Language = Language.UNKNOWN
    category: FileCategory = FileCategory.UNKNOWN

    @property
    def match_types(self) -> List[str]:
        """Get list of match types for this file"""
        types = []
        if self.filename_matched:
            types.append("filename")
        if self.content_matches:
            types.append("content")
        if self.symbol_matches:
            types.append("symbol")
        return types


class HybridDiscoveryEngine:
    """
    Intelligent file discovery using multiple search strategies + LLM reasoning.

    Combines:
    1. Filename pattern matching
    2. Content search (ripgrep)
    3. Symbol finding (functions/classes)
    4. LLM reasoning for relevance ranking
    """

    def __init__(
        self,
        scanner: RepositoryScanner,
        searcher: CodeSearcher,
        symbol_finder: SymbolFinder,
        llm_provider: Optional[ILLMProvider] = None
    ):
        """
        Initialize discovery engine.

        Args:
            scanner: Repository scanner for file tree
            searcher: Code searcher for content search
            symbol_finder: Symbol finder for function/class search
            llm_provider: Optional LLM for intelligent ranking
        """
        self.scanner = scanner
        self.searcher = searcher
        self.symbol_finder = symbol_finder
        self.llm = llm_provider

    async def discover_files(
        self,
        query: str,
        max_files: int = 10,
        strategy: DiscoveryStrategy = DiscoveryStrategy.COMBINED,
        use_llm_ranking: bool = True,
        include_snippets: bool = True
    ) -> List[FileDiscoveryResult]:
        """
        Discover relevant files for a query.

        Args:
            query: Natural language query
            max_files: Maximum files to return
            strategy: Search strategy to use
            use_llm_ranking: Whether to use LLM for ranking
            include_snippets: Whether to include code snippets

        Returns:
            List of discovered files with relevance scores
        """
        # Step 1: Execute search strategies
        filename_matches = {}
        content_matches = {}
        symbol_matches = {}

        if strategy in (DiscoveryStrategy.FILENAME, DiscoveryStrategy.COMBINED):
            filename_matches = await self._search_filenames(query)

        if strategy in (DiscoveryStrategy.CONTENT, DiscoveryStrategy.COMBINED):
            content_matches = await self._search_content(query)

        if strategy in (DiscoveryStrategy.SYMBOL, DiscoveryStrategy.COMBINED):
            symbol_matches = await self._search_symbols(query)

        # Step 2: Aggregate results by file
        file_matches = self._aggregate_results(
            filename_matches,
            content_matches,
            symbol_matches
        )

        if not file_matches:
            return []

        # Step 3: Rank with LLM (if enabled and available)
        if use_llm_ranking and self.llm:
            results = await self._rank_with_llm(query, file_matches, max_files)
        else:
            results = self._rank_with_heuristics(query, file_matches, max_files)

        # Step 4: Add snippets if requested
        if include_snippets:
            for result in results:
                if not result.snippet:
                    match_data = file_matches.get(result.file_path)
                    if match_data:
                        result.snippet = self._extract_snippet(
                            result.file_path,
                            match_data.content_matches + match_data.symbol_matches
                        )

        return results[:max_files]

    async def _search_filenames(self, query: str) -> Dict[str, int]:
        """
        Search for files by name pattern.

        Args:
            query: Search query

        Returns:
            Dict mapping file path to match count (1 for filename match)
        """
        # Extract potential filename keywords from query
        # e.g., "authentication logic" -> search for "*auth*"
        keywords = self._extract_filename_keywords(query)

        matches = {}
        for keyword in keywords:
            # Search using scanner
            pattern = f"*{keyword}*"
            results = await self.scanner.search_files(pattern, case_sensitive=False)

            for node in results:
                if node.path not in matches:
                    matches[node.path] = 1

        return matches

    async def _search_content(self, query: str) -> Dict[str, List[SearchResult]]:
        """
        Search file contents.

        Args:
            query: Search query

        Returns:
            Dict mapping file path to list of search results
        """
        # Use code searcher to find content matches
        results = self.searcher.search_text(query, max_results=100)

        # Group by file
        by_file = defaultdict(list)
        for result in results:
            by_file[result.file_path].append(result)

        return dict(by_file)

    async def _search_symbols(self, query: str) -> Dict[str, List[SearchResult]]:
        """
        Search for symbol definitions.

        Args:
            query: Search query (symbol name)

        Returns:
            Dict mapping file path to list of search results
        """
        # Extract potential symbol names from query
        # e.g., "UserController class" -> search for "UserController"
        symbols = self._extract_symbol_names(query)

        by_file = defaultdict(list)
        for symbol in symbols:
            # Search for symbol
            results = self.symbol_finder.find_symbol(
                symbol_name=symbol,
                exact_match=False
            )

            for result in results:
                by_file[result.file_path].append(result)

        return dict(by_file)

    def _aggregate_results(
        self,
        filename_matches: Dict[str, int],
        content_matches: Dict[str, List[SearchResult]],
        symbol_matches: Dict[str, List[SearchResult]]
    ) -> Dict[str, FileMatchData]:
        """
        Aggregate all search results by file.

        Args:
            filename_matches: Files matched by name
            content_matches: Files matched by content
            symbol_matches: Files matched by symbol

        Returns:
            Dict mapping file path to aggregated match data
        """
        all_files = set()
        all_files.update(filename_matches.keys())
        all_files.update(content_matches.keys())
        all_files.update(symbol_matches.keys())

        aggregated = {}
        for file_path in all_files:
            content_results = content_matches.get(file_path, [])
            symbol_results = symbol_matches.get(file_path, [])

            # Get language and category from first result
            language = Language.UNKNOWN
            category = FileCategory.UNKNOWN
            if content_results:
                language = content_results[0].language
                category = content_results[0].category
            elif symbol_results:
                language = symbol_results[0].language
                category = symbol_results[0].category

            match_data = FileMatchData(
                file_path=file_path,
                filename_matched=file_path in filename_matches,
                content_matches=content_results,
                symbol_matches=symbol_results,
                total_matches=len(content_results) + len(symbol_results),
                language=language,
                category=category
            )

            aggregated[file_path] = match_data

        return aggregated

    async def _rank_with_llm(
        self,
        query: str,
        file_matches: Dict[str, FileMatchData],
        max_files: int
    ) -> List[FileDiscoveryResult]:
        """
        Use LLM to rank files and provide reasoning.

        Args:
            query: Original query
            file_matches: Aggregated match data
            max_files: Maximum files to return

        Returns:
            Ranked list of file discovery results
        """
        # Build prompt with file information
        file_list = []
        for file_path, match_data in list(file_matches.items())[:20]:  # Limit to top 20
            snippet = self._extract_snippet(
                file_path,
                match_data.content_matches + match_data.symbol_matches
            )

            file_info = {
                "file_path": file_path,
                "match_count": match_data.total_matches,
                "match_types": match_data.match_types,
                "snippet": snippet[:200] if snippet else "",  # Truncate
                "language": match_data.language.value,
                "category": match_data.category.value
            }
            file_list.append(file_info)

        if not file_list:
            return []

        prompt = self._build_ranking_prompt(query, file_list, max_files)

        # Make LLM request
        try:
            response = await self.llm.create_message(
                system_prompt="You are a code analysis assistant helping discover relevant files.",
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                temperature=0.3,
                max_tokens=2000
            )

            # Parse LLM response
            results = self._parse_llm_ranking(response.content, file_matches)
            return results

        except Exception as e:
            # Fallback to heuristic ranking if LLM fails
            logger.warning(f"LLM ranking failed: {e}, falling back to heuristics")
            return self._rank_with_heuristics(query, file_matches, max_files)

    def _rank_with_heuristics(
        self,
        query: str,
        file_matches: Dict[str, FileMatchData],
        max_files: int
    ) -> List[FileDiscoveryResult]:
        """
        Rank files using heuristic scoring.

        Args:
            query: Original query
            file_matches: Aggregated match data
            max_files: Maximum files to return

        Returns:
            Ranked list of file discovery results
        """
        results = []
        query_lower = query.lower()

        for file_path, match_data in file_matches.items():
            score = 0.0

            # Filename match bonus
            if match_data.filename_matched:
                score += 0.3

            # Match count (normalized)
            score += min(match_data.total_matches * 0.1, 0.4)

            # Symbol match bonus (definitions are important)
            if match_data.symbol_matches:
                score += 0.3

            # File category bonus
            if match_data.category == FileCategory.SOURCE_CODE:
                score += 0.2
            elif match_data.category == FileCategory.TEST:
                score += 0.1

            # Query keyword in path
            path_lower = file_path.lower()
            for word in query_lower.split():
                if len(word) > 3 and word in path_lower:
                    score += 0.1

            # Normalize score to 0-1
            score = min(score, 1.0)

            # Extract snippet
            snippet = self._extract_snippet(
                file_path,
                match_data.content_matches + match_data.symbol_matches
            )

            result = FileDiscoveryResult(
                file_path=file_path,
                relevance_score=score,
                match_count=match_data.total_matches,
                snippet=snippet,
                reasoning=f"Found {match_data.total_matches} matches ({', '.join(match_data.match_types)})",
                match_types=match_data.match_types,
                language=match_data.language,
                category=match_data.category
            )
            results.append(result)

        # Sort by score
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:max_files]

    def _build_ranking_prompt(
        self,
        query: str,
        file_list: List[Dict],
        max_files: int
    ) -> str:
        """Build prompt for LLM ranking"""
        files_json = json.dumps(file_list, indent=2)

        return f"""You are helping discover relevant files in a codebase.

User Query: "{query}"

I have found the following files that might be relevant:

{files_json}

Please analyze these files and rank them by relevance to the query.
For each relevant file, provide:
1. A relevance score (0.0 to 1.0, where 1.0 is most relevant)
2. A brief explanation of why it's relevant

Return ONLY a JSON array of the top {max_files} most relevant files in this format:
[
  {{
    "file_path": "path/to/file.py",
    "relevance_score": 0.95,
    "reasoning": "Contains the main authentication logic"
  }}
]

Only include files that are actually relevant to the query. Exclude false positives.
Return valid JSON only, no other text."""

    def _parse_llm_ranking(
        self,
        llm_response: str,
        file_matches: Dict[str, FileMatchData]
    ) -> List[FileDiscoveryResult]:
        """Parse LLM ranking response into FileDiscoveryResult objects"""
        try:
            # Extract JSON from response
            # Look for JSON array in the response
            start = llm_response.find('[')
            end = llm_response.rfind(']') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")

            json_str = llm_response[start:end]
            rankings = json.loads(json_str)

            results = []
            for item in rankings:
                file_path = item.get("file_path")
                if file_path not in file_matches:
                    continue

                match_data = file_matches[file_path]
                snippet = self._extract_snippet(
                    file_path,
                    match_data.content_matches + match_data.symbol_matches
                )

                result = FileDiscoveryResult(
                    file_path=file_path,
                    relevance_score=float(item.get("relevance_score", 0.5)),
                    match_count=match_data.total_matches,
                    snippet=snippet,
                    reasoning=item.get("reasoning", "Relevant to query"),
                    match_types=match_data.match_types,
                    language=match_data.language,
                    category=match_data.category
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []

    def _extract_snippet(
        self,
        file_path: str,
        matches: List[SearchResult],
        max_length: int = 200
    ) -> str:
        """
        Extract most relevant snippet from matches.

        Args:
            file_path: Path to file
            matches: List of search results
            max_length: Maximum snippet length

        Returns:
            Code snippet
        """
        if not matches:
            return ""

        # Use the match with highest relevance score
        best_match = max(matches, key=lambda m: m.relevance_score)

        # Build snippet with context
        lines = []
        if best_match.context_before:
            lines.extend(best_match.context_before[-1:])  # 1 line before
        lines.append(best_match.line_content)
        if best_match.context_after:
            lines.extend(best_match.context_after[:2])  # 2 lines after

        snippet = "\n".join(lines)

        # Truncate if too long
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."

        return snippet

    def _extract_filename_keywords(self, query: str) -> List[str]:
        """
        Extract potential filename keywords from query.

        Args:
            query: User query

        Returns:
            List of filename keywords
        """
        # Common words to skip
        skip_words = {
            "and", "or", "the", "a", "an", "in", "on", "at", "to", "for",
            "with", "by", "from", "of", "file", "files", "code", "logic",
            "implementation", "function", "class", "module"
        }

        words = query.lower().split()
        keywords = []

        for word in words:
            # Remove punctuation
            word = word.strip('.,!?;:()[]{}')

            # Skip short words and common words
            if len(word) > 3 and word not in skip_words:
                keywords.append(word)

        return keywords[:5]  # Limit to top 5

    def _extract_symbol_names(self, query: str) -> List[str]:
        """
        Extract potential symbol names from query.

        Args:
            query: User query

        Returns:
            List of potential symbol names
        """
        # Look for capitalized words (potential class names)
        # Look for words ending in common function patterns
        words = query.split()
        symbols = []

        for word in words:
            # Remove punctuation
            word = word.strip('.,!?;:()[]{}')

            # Capitalized words (potential class names)
            if word and word[0].isupper() and len(word) > 2:
                symbols.append(word)

            # Snake_case or camelCase identifiers
            if "_" in word or (any(c.isupper() for c in word[1:]) and len(word) > 3):
                symbols.append(word)

        return symbols[:3]  # Limit to top 3
