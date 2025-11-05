"""
Repository structure caching.

Provides caching functionality to avoid repeated GitHub API calls
for repository tree information.
"""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class CacheEntry:
    """Cache entry with data and metadata"""
    data: Any
    timestamp: float
    commit_sha: str
    size_bytes: int

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.timestamp) > ttl_seconds

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class RepositoryCache:
    """
    In-memory cache for repository tree structures.

    Caches repository file trees to reduce GitHub API calls.
    Uses commit SHA for cache invalidation.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size_mb: int = 50):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            max_size_mb: Maximum cache size in megabytes
        """
        self.ttl_seconds = ttl_seconds
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, owner: str, repo: str, ref: str, commit_sha: str) -> str:
        """Generate cache key."""
        return f"{owner}/{repo}/{ref}/{commit_sha}"

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            # Convert to JSON string to estimate size
            json_str = json.dumps(data)
            return len(json_str.encode('utf-8'))
        except:
            return 0

    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(entry.size_bytes for entry in self._cache.values())

    def _evict_if_needed(self, new_size: int):
        """Evict oldest entries if cache would exceed max size."""
        current_size = self._get_total_size()

        if current_size + new_size <= self.max_size_bytes:
            return

        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )

        # Remove oldest entries until we have space
        for key, entry in sorted_entries:
            if current_size + new_size <= self.max_size_bytes:
                break
            current_size -= entry.size_bytes
            del self._cache[key]

    def get(
        self,
        owner: str,
        repo: str,
        ref: str,
        commit_sha: str
    ) -> Optional[Any]:
        """
        Get cached data.

        Args:
            owner: Repository owner
            repo: Repository name
            ref: Branch/tag/ref name
            commit_sha: Commit SHA

        Returns:
            Cached data or None if not found/expired
        """
        key = self._make_key(owner, repo, ref, commit_sha)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check if expired
        if entry.is_expired(self.ttl_seconds):
            del self._cache[key]
            self._misses += 1
            return None

        # Check if commit SHA matches (cache invalidation)
        if entry.commit_sha != commit_sha:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.data

    def set(
        self,
        owner: str,
        repo: str,
        ref: str,
        commit_sha: str,
        data: Any
    ):
        """
        Store data in cache.

        Args:
            owner: Repository owner
            repo: Repository name
            ref: Branch/tag/ref name
            commit_sha: Commit SHA
            data: Data to cache
        """
        key = self._make_key(owner, repo, ref, commit_sha)
        size = self._estimate_size(data)

        # Evict if needed
        self._evict_if_needed(size)

        # Store entry
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            commit_sha=commit_sha,
            size_bytes=size
        )
        self._cache[key] = entry

    def invalidate(
        self,
        owner: str,
        repo: str,
        ref: Optional[str] = None
    ):
        """
        Invalidate cache entries.

        Args:
            owner: Repository owner
            repo: Repository name
            ref: Optional specific ref to invalidate (if None, invalidates all refs)
        """
        prefix = f"{owner}/{repo}/"
        if ref:
            prefix += f"{ref}/"

        keys_to_delete = [
            key for key in self._cache.keys()
            if key.startswith(prefix)
        ]

        for key in keys_to_delete:
            del self._cache[key]

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "size_bytes": self._get_total_size(),
            "size_mb": round(self._get_total_size() / (1024 * 1024), 2),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "ttl_seconds": self.ttl_seconds
        }
