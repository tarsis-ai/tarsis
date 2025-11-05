"""
GitHub API integration for Tarsis.

Provides a clean interface for GitHub operations needed for issue implementation.
"""

from .client import (
    GitHubClient,
    GitHubConfig,
    IssueDetails,
    PullRequestDetails,
    GitHubAPIError,
    GitHubNotFoundError
)

__all__ = [
    "GitHubClient",
    "GitHubConfig",
    "IssueDetails",
    "PullRequestDetails",
    "GitHubAPIError",
    "GitHubNotFoundError"
]
