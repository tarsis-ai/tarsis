# Code Search Module

Fast code search and symbol finding for Tarsis using ripgrep.

## Overview

The search module provides powerful code search capabilities including:

- **Content Search** - Search for text within files using ripgrep
- **Symbol Finding** - Locate function/class definitions across languages
- **Pattern Matching** - Advanced regex search with ranking
- **Relevance Scoring** - Intelligent result ranking

## Components

### CodeSearcher

Core search engine that wraps ripgrep for fast code search.

```python
from tarsis.repository import CodeSearcher, SearchOptions, SearchType

searcher = CodeSearcher("/path/to/repo")

# Simple text search
results = searcher.search_text("authenticate", file_pattern="*.py")

# Regex search
results = searcher.search_regex(r"class\s+\w+Controller", file_pattern="*.py")

# Advanced search with options
options = SearchOptions(
    query="login",
    search_type=SearchType.TEXT,
    case_sensitive=False,
    file_pattern="*.py",
    context_lines=3,
    max_results=50
)
results = searcher.search(options)
```

### SymbolFinder

Finds code symbol definitions (functions, classes, etc.) across multiple languages.

```python
from tarsis.repository import SymbolFinder, SymbolType, Language

searcher = CodeSearcher("/path/to/repo")
finder = SymbolFinder(searcher)

# Find any symbol
results = finder.find_symbol("UserController")

# Find specific symbol type
results = finder.find_symbol("login", symbol_type=SymbolType.FUNCTION)

# Find in specific language
results = finder.find_symbol("authenticate", language=Language.PYTHON)

# Convenience methods
results = finder.find_function("process_payment")
results = finder.find_class("UserModel")
results = finder.find_imports("flask")
```

### ResultRanker

Ranks search results by relevance based on multiple factors:

- Match quality (exact vs partial)
- Match location (definition vs usage)
- File importance (source > test > docs)
- Position in line
- Line length (penalizes minified code)

Results are automatically ranked when using `CodeSearcher.search()`.

## Supported Languages

Symbol finding supports:

- Python
- JavaScript/TypeScript
- Go
- Java
- Rust
- C#
- C/C++
- Ruby
- PHP

## Search Result Format

```python
@dataclass
class SearchResult:
    file_path: str              # Path to file
    line_number: int            # Line number (1-indexed)
    line_content: str           # The matched line
    match_start: int            # Start position of match
    match_end: int              # End position of match
    context_before: List[str]   # Lines before match
    context_after: List[str]    # Lines after match
    relevance_score: float      # Relevance score (higher = more relevant)
    language: Language          # Detected language
    category: FileCategory      # File category
    symbol_type: SymbolType     # Type of symbol (if symbol search)
```

## Integration with Tools

The search module is integrated into Tarsis as three tools:

### search_code

Search for text within file contents.

```json
{
  "tool": "search_code",
  "input": {
    "query": "authenticate",
    "file_pattern": "*.py",
    "context_lines": 2,
    "max_results": 30
  }
}
```

### find_symbol

Find function/class definitions.

```json
{
  "tool": "find_symbol",
  "input": {
    "symbol_name": "UserController",
    "symbol_type": "class",
    "language": "python"
  }
}
```

### grep_pattern

Advanced regex pattern search.

```json
{
  "tool": "grep_pattern",
  "input": {
    "pattern": "def\\s+test_\\w+",
    "file_pattern": "test_*.py",
    "sort_by": "relevance"
  }
}
```

## Requirements

- **ripgrep** - Fast search tool (https://github.com/BurntSushi/ripgrep)
- **GitPython** - For repository cloning
- Python 3.10+

### Installing ripgrep

```bash
# Ubuntu/Debian
sudo apt install ripgrep

# Fedora
sudo dnf install ripgrep

# macOS
brew install ripgrep

# Windows
choco install ripgrep
```

## Local Repository Cloning

The search module automatically clones repositories locally for fast searching:

1. First search triggers a shallow clone (depth=1)
2. Repository is cached in temporary directory
3. Subsequent searches reuse the cached clone
4. Clone is updated if repository changes

### Clone Location

Repositories are cloned to: `/tmp/tarsis_repo_<timestamp>/<repo_name>`

## Performance

- **Fast** - ripgrep is highly optimized (C with SIMD)
- **Parallel** - Searches use all available CPU cores
- **Cached** - Repository clones are reused
- **Filtered** - Automatically excludes .git, node_modules, etc.

### Benchmarks

On a typical repository (10K files):

- Content search: ~100-500ms
- Symbol finding: ~200-800ms
- Pattern matching: ~150-600ms

## Examples

### Example 1: Find Authentication Code

```python
searcher = CodeSearcher("/path/to/repo")

# Search for authentication-related code
results = searcher.search_text("authenticate", file_pattern="*.py")

for result in results[:5]:
    print(f"{result.file_path}:{result.line_number}")
    print(f"  Score: {result.relevance_score:.2f}")
    print(f"  {result.line_content}")
```

### Example 2: Find All Test Functions

```python
finder = SymbolFinder(searcher)

# Find all test functions
results = finder.find_symbol("test_", symbol_type=SymbolType.FUNCTION, exact_match=False)

for result in results:
    print(f"Test: {result.line_content.strip()}")
    print(f"  in {result.file_path}:{result.line_number}")
```

### Example 3: Find Import Usage

```python
# Find all files importing 'requests'
results = finder.find_imports("requests", language=Language.PYTHON)

for result in results:
    print(f"{result.file_path}: {result.line_content.strip()}")
```

### Example 4: Complex Regex Search

```python
# Find all class definitions that inherit from BaseModel
options = SearchOptions(
    query=r"class\s+\w+\(.*BaseModel.*\)",
    search_type=SearchType.REGEX,
    file_pattern="*.py",
    max_results=100
)

results = searcher.search(options)

for result in results:
    print(f"Model class found: {result.file_path}:{result.line_number}")
```

## Architecture

```
┌─────────────────────────────────────┐
│  Search Tools (tools/search_tools.py)│
│  - SearchCodeHandler                 │
│  - FindSymbolHandler                 │
│  - GrepPatternHandler                │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  Search Module (repository/search.py)│
│  - CodeSearcher (ripgrep wrapper)    │
│  - SymbolFinder (symbol extraction)  │
│  - ResultRanker (relevance scoring)  │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  ripgrep (external binary)           │
│  - Fast regex search engine          │
│  - JSON output format                │
└──────────────────────────────────────┘
```

## Error Handling

The search module handles various error conditions:

- **ripgrep not installed** - Raises RuntimeError with installation instructions
- **Repository clone failed** - Provides detailed error message
- **Search timeout** - 30 second timeout on searches
- **Invalid regex** - Returns error response from ripgrep
- **No results** - Returns empty list (not an error)

## Future Enhancements

Potential improvements for future versions:

- [ ] Incremental repository updates (git pull instead of full clone)
- [ ] Search result caching
- [ ] Fuzzy symbol matching
- [ ] Cross-reference analysis (find all usages of a symbol)
- [ ] Semantic search integration (vector embeddings)
- [ ] Search history and suggestions
- [ ] Parallel multi-repo search

## References

- ripgrep: https://github.com/BurntSushi/ripgrep
- GitPython: https://gitpython.readthedocs.io/
- Language regex patterns inspired by tree-sitter grammars

---

**Version**: 1.0.0
**Last Updated**: 2025-10-29
**Status**: Production Ready ✅
