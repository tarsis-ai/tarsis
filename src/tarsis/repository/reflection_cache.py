"""
Repository Reflection Cache - Persistent storage for cross-issue learning

Enables the agent to learn from past issues in the same repository by:
- Saving reflections to disk after each issue
- Loading relevant reflections when starting new issues
- Finding similar past reflections using keyword matching
- Cleaning up old reflections automatically

This implements repository-level learning as described in the Reflexion framework.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ReflectionCache:
    """
    Persistent storage for reflections across GitHub issues.

    Reflections are stored as JSON files in a directory structure:
    {cache_dir}/{repo_owner}/{repo_name}/issue_{number}.json

    Each file contains reflections from one issue, along with metadata.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize reflection cache.

        Args:
            cache_dir: Base directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Reflection cache initialized at: {self.cache_dir}")

    def save_reflections(
        self,
        repo_owner: str,
        repo_name: str,
        issue_number: str,
        reflections: List[Any]
    ) -> None:
        """
        Save reflections to repository cache.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            issue_number: Issue number
            reflections: List of ReflectionEntry objects
        """
        try:
            # Create repository cache directory
            repo_cache_dir = self.cache_dir / repo_owner / repo_name
            repo_cache_dir.mkdir(parents=True, exist_ok=True)

            # Cache file path
            cache_file = repo_cache_dir / f"issue_{issue_number}.json"

            # Serialize reflections
            data = {
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "issue_number": issue_number,
                "timestamp": datetime.now().isoformat(),
                "reflection_count": len(reflections),
                "reflections": [self._serialize_entry(r) for r in reflections]
            }

            # Write to file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Saved {len(reflections)} reflections to {cache_file}")

        except Exception as e:
            logger.error(f"Failed to save reflections to cache: {e}", exc_info=True)

    def load_reflections(
        self,
        repo_owner: str,
        repo_name: str,
        max_age_days: int = 30
    ) -> List[Any]:
        """
        Load recent reflections from repository cache.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            max_age_days: Maximum age of reflections to load (in days)

        Returns:
            List of ReflectionEntry objects
        """
        try:
            repo_cache_dir = self.cache_dir / repo_owner / repo_name

            if not repo_cache_dir.exists():
                logger.debug(f"No cache directory found for {repo_owner}/{repo_name}")
                return []

            all_reflections = []
            cutoff_date = datetime.now() - timedelta(days=max_age_days)

            # Load from all recent cache files
            for cache_file in repo_cache_dir.glob("issue_*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Check file age
                    file_date = datetime.fromisoformat(data["timestamp"])
                    if file_date < cutoff_date:
                        logger.debug(f"Skipping old cache file: {cache_file.name}")
                        continue

                    # Deserialize reflections
                    for entry_data in data["reflections"]:
                        entry = self._deserialize_entry(entry_data)
                        all_reflections.append(entry)

                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    continue

            logger.info(f"📚 Loaded {len(all_reflections)} reflections from cache ({repo_owner}/{repo_name})")
            return all_reflections

        except Exception as e:
            logger.error(f"Failed to load reflections from cache: {e}", exc_info=True)
            return []

    def get_similar_reflections(
        self,
        repo_owner: str,
        repo_name: str,
        query_context: Dict[str, Any],
        limit: int = 5
    ) -> List[Any]:
        """
        Find reflections similar to current context.

        Uses simple keyword-based similarity matching. Can be extended
        with embedding-based similarity in the future.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            query_context: Context to match against (trigger, error types, etc.)
            limit: Maximum number of reflections to return

        Returns:
            List of similar ReflectionEntry objects, sorted by relevance
        """
        all_reflections = self.load_reflections(repo_owner, repo_name)

        if not all_reflections:
            return []

        # Score reflections by similarity
        scored = []
        for reflection in all_reflections:
            score = self._compute_similarity(reflection, query_context)
            scored.append((score, reflection))

        # Sort by score (descending) and return top N
        scored.sort(reverse=True, key=lambda x: x[0])
        similar = [r for _, r in scored[:limit] if _ > 0]

        logger.debug(f"Found {len(similar)} similar reflections (threshold > 0)")
        return similar

    def cleanup_old_reflections(
        self,
        repo_owner: str,
        repo_name: str,
        max_age_days: int = 90
    ) -> int:
        """
        Remove old reflection cache files.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            max_age_days: Maximum age to keep (in days)

        Returns:
            Number of files removed
        """
        try:
            repo_cache_dir = self.cache_dir / repo_owner / repo_name

            if not repo_cache_dir.exists():
                return 0

            removed = 0
            cutoff_date = datetime.now() - timedelta(days=max_age_days)

            for cache_file in repo_cache_dir.glob("issue_*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    file_date = datetime.fromisoformat(data["timestamp"])
                    if file_date < cutoff_date:
                        cache_file.unlink()
                        removed += 1
                        logger.debug(f"Removed old cache file: {cache_file.name}")

                except Exception as e:
                    logger.warning(f"Failed to process cache file {cache_file}: {e}")
                    continue

            if removed > 0:
                logger.info(f"🧹 Cleaned up {removed} old reflection cache files")

            return removed

        except Exception as e:
            logger.error(f"Failed to cleanup old reflections: {e}", exc_info=True)
            return 0

    def _compute_similarity(
        self,
        reflection: Any,
        query_context: Dict[str, Any]
    ) -> float:
        """
        Compute similarity score between reflection and query context.

        Uses simple keyword matching. Score ranges from 0.0 to ~5.0.

        Args:
            reflection: ReflectionEntry object
            query_context: Context dict with trigger, errors, etc.

        Returns:
            Similarity score (higher = more similar)
        """
        score = 0.0

        # Same trigger type: +1.0
        if hasattr(reflection, 'trigger'):
            trigger_value = reflection.trigger.value if hasattr(reflection.trigger, 'value') else str(reflection.trigger)
            query_trigger = query_context.get("trigger")
            if query_trigger and trigger_value == query_trigger:
                score += 1.0

        # Keyword matching in insight text
        if hasattr(reflection, 'insight'):
            query_text = str(query_context).lower()
            reflection_text = reflection.insight.lower()

            # Common error keywords
            error_keywords = [
                "test", "validation", "import", "syntax", "type", "error",
                "file", "missing", "not found", "failed", "exception"
            ]

            for keyword in error_keywords:
                if keyword in query_text and keyword in reflection_text:
                    score += 0.5

            # Tool names
            tool_keywords = [
                "modify_file", "commit_changes", "run_validation",
                "create_branch", "create_pull_request", "read_file"
            ]

            for tool in tool_keywords:
                if tool in query_text and tool in reflection_text:
                    score += 0.3

        # Context similarity (if available)
        if hasattr(reflection, 'context'):
            query_files = set(query_context.get("files_modified", []))
            reflection_files = set(reflection.context.get("files_modified", []))

            # File overlap
            if query_files and reflection_files:
                overlap = len(query_files & reflection_files)
                if overlap > 0:
                    score += overlap * 0.2

        return score

    @staticmethod
    def _serialize_entry(entry: Any) -> Dict:
        """
        Convert ReflectionEntry to JSON-serializable dict.

        Args:
            entry: ReflectionEntry object

        Returns:
            Dict representation
        """
        # Handle both ReflectionEntry objects and dicts
        if hasattr(entry, '__dict__'):
            data = {
                "iteration": getattr(entry, 'iteration', 0),
                "trigger": getattr(entry, 'trigger').value if hasattr(getattr(entry, 'trigger', None), 'value') else str(getattr(entry, 'trigger', 'unknown')),
                "context": getattr(entry, 'context', {}),
                "insight": getattr(entry, 'insight', ''),
                "timestamp": getattr(entry, 'timestamp', datetime.now().isoformat()),
                "applied": getattr(entry, 'applied', False)
            }
        else:
            # Already a dict
            data = entry

        return data

    @staticmethod
    def _deserialize_entry(data: Dict) -> Any:
        """
        Convert dict back to ReflectionEntry.

        Args:
            data: Dict representation

        Returns:
            ReflectionEntry object
        """
        try:
            from ..agent.reflection import ReflectionEntry, ReflectionTrigger

            return ReflectionEntry(
                iteration=data.get("iteration", 0),
                trigger=ReflectionTrigger(data.get("trigger", "periodic")),
                context=data.get("context", {}),
                insight=data.get("insight", ""),
                timestamp=data.get("timestamp", datetime.now().isoformat()),
                applied=data.get("applied", False)
            )
        except Exception as e:
            logger.warning(f"Failed to deserialize reflection entry: {e}")
            # Return a minimal entry on error
            from ..agent.reflection import ReflectionEntry, ReflectionTrigger
            return ReflectionEntry(
                iteration=0,
                trigger=ReflectionTrigger.PERIODIC,
                context={},
                insight=data.get("insight", "Failed to load reflection"),
                timestamp=datetime.now().isoformat(),
                applied=False
            )

    def get_cache_stats(self, repo_owner: str, repo_name: str) -> Dict[str, Any]:
        """
        Get statistics about cached reflections for a repository.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name

        Returns:
            Dict with cache statistics
        """
        try:
            repo_cache_dir = self.cache_dir / repo_owner / repo_name

            if not repo_cache_dir.exists():
                return {
                    "cache_exists": False,
                    "total_files": 0,
                    "total_reflections": 0
                }

            files = list(repo_cache_dir.glob("issue_*.json"))
            total_reflections = 0

            for cache_file in files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        total_reflections += data.get("reflection_count", 0)
                except:
                    continue

            return {
                "cache_exists": True,
                "total_files": len(files),
                "total_reflections": total_reflections,
                "cache_path": str(repo_cache_dir)
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
