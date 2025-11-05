"""
File discovery tool handlers.

These tools allow the agent to intelligently discover relevant files
using hybrid search strategies and LLM reasoning.
"""

import os
import logging
from typing import Dict, Any, Optional
from .base import BaseToolHandler, ToolDefinition, ToolResponse, ToolCategory
from ..repository import (
    HybridDiscoveryEngine,
    DiscoveryStrategy,
    CodeSearcher,
    SymbolFinder,
    RepositoryScanner
)
from ..github import GitHubClient, GitHubConfig
from ..llm import create_llm_provider
import subprocess
import tempfile


logger = logging.getLogger(__name__)

# Global instances
_discovery_engine: Optional[HybridDiscoveryEngine] = None
_repo_clone_path: Optional[str] = None


def _get_or_clone_repository() -> str:
    """
    Get or clone the repository locally for searching.

    Returns:
        Path to local repository
    """
    global _repo_clone_path

    if _repo_clone_path and os.path.exists(_repo_clone_path):
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


def _get_discovery_engine() -> HybridDiscoveryEngine:
    """Get or create the discovery engine instance."""
    global _discovery_engine

    if _discovery_engine is None:
        # Get GitHub client
        config = GitHubConfig(
            token=os.getenv("GITHUB_TOKEN", ""),
            repo_owner=os.getenv("GITHUB_REPO_OWNER", ""),
            repo_name=os.getenv("GITHUB_REPO_NAME", "")
        )
        github = GitHubClient(config)

        # Get repository scanner
        scanner = RepositoryScanner(github)

        # Get code searcher
        repo_path = _get_or_clone_repository()
        searcher = CodeSearcher(repo_path)

        # Get symbol finder
        symbol_finder = SymbolFinder(searcher)

        # Get LLM provider
        llm_provider_type = os.getenv("LLM_PROVIDER", "ollama")
        llm_model = os.getenv("LLM_MODEL_ID")
        llm_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

        try:
            llm_provider = create_llm_provider(
                provider_type=llm_provider_type,
                model_id=llm_model,
                api_key=llm_api_key
            )
        except Exception as e:
            logger.warning(f"Could not create LLM provider for discovery: {e}")
            llm_provider = None

        # Create discovery engine
        _discovery_engine = HybridDiscoveryEngine(
            scanner=scanner,
            searcher=searcher,
            symbol_finder=symbol_finder,
            llm_provider=llm_provider
        )

    return _discovery_engine


class DiscoverRelevantFilesHandler(BaseToolHandler):
    """Tool to intelligently discover relevant files using hybrid search + LLM reasoning"""

    @property
    def name(self) -> str:
        return "discover_relevant_files"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE_ANALYSIS

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Intelligently discover files relevant to a query using multiple search "
                "strategies and LLM reasoning. Returns a ranked list of FILES (not individual "
                "matches) with explanations of why they're relevant. This is more efficient than "
                "searching and manually filtering results. Use this when you need to understand "
                "which files to read for a specific task, feature, or concept. "
                "Examples: 'authentication and login logic', 'payment processing', "
                "'error handling utilities', 'database models'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language query describing what you're looking for. "
                            "Be specific about the functionality, feature, or concept. "
                            "Examples: 'authentication and login logic', 'payment processing with Stripe', "
                            "'user registration and validation', 'API endpoint definitions'"
                        )
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum number of files to return (default: 10, max: 20)",
                        "default": 10
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["filename", "content", "symbol", "combined"],
                        "description": (
                            "Search strategy to use:\n"
                            "- 'filename': Fast, searches only filenames\n"
                            "- 'content': Comprehensive, searches file contents\n"
                            "- 'symbol': Finds function/class definitions\n"
                            "- 'combined': Uses all strategies (default, recommended)"
                        ),
                        "default": "combined"
                    },
                    "use_llm_ranking": {
                        "type": "boolean",
                        "description": (
                            "Whether to use LLM for intelligent ranking (default: true). "
                            "LLM ranking provides better relevance and explanations but is slower."
                        ),
                        "default": True
                    }
                },
                "required": ["query"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Discover relevant files"""
        try:
            query = input_data["query"]
            max_files = min(input_data.get("max_files", 10), 20)  # Cap at 20
            strategy_str = input_data.get("strategy", "combined")
            use_llm_ranking = input_data.get("use_llm_ranking", True)

            # Parse strategy
            try:
                strategy = DiscoveryStrategy(strategy_str)
            except ValueError:
                return self._error_response(
                    ValueError(f"Invalid strategy: {strategy_str}")
                )

            # Get discovery engine
            engine = _get_discovery_engine()

            # Discover files
            results = await engine.discover_files(
                query=query,
                max_files=max_files,
                strategy=strategy,
                use_llm_ranking=use_llm_ranking,
                include_snippets=True
            )

            if not results:
                return self._success_response(
                    f"No relevant files found for query: '{query}'",
                    metadata={"count": 0, "query": query}
                )

            # Format results
            result_lines = [
                f"Discovered {len(results)} relevant file(s) for '{query}':\n"
            ]

            for i, result in enumerate(results, 1):
                result_lines.append(f"\n{i}. {result.file_path}")
                result_lines.append(f"   Relevance: {result.relevance_score:.2f}/1.0")
                result_lines.append(f"   Reason: {result.reasoning}")
                result_lines.append(f"   Matches: {result.match_count} ({', '.join(result.match_types)})")
                result_lines.append(f"   Language: {result.language.value}")

                if result.snippet:
                    result_lines.append(f"   Preview:")
                    for line in result.snippet.split('\n')[:3]:  # First 3 lines
                        result_lines.append(f"      {line}")

            # Add usage hint
            result_lines.append(f"\nTip: Use read_file tool to examine these files in detail.")

            result_text = "\n".join(result_lines)

            return self._success_response(
                result_text,
                metadata={
                    "count": len(results),
                    "query": query,
                    "strategy": strategy_str,
                    "files": [
                        {
                            "path": r.file_path,
                            "score": r.relevance_score,
                            "matches": r.match_count,
                            "reasoning": r.reasoning
                        }
                        for r in results
                    ]
                }
            )

        except Exception as e:
            return self._error_response(e)
