"""
GitHub API client for Tarsis.

Provides async access to GitHub API operations needed for issue implementation.
"""

import httpx
import base64
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    """GitHub API configuration"""
    token: str
    repo_owner: str
    repo_name: str
    api_url: str = "https://api.github.com"
    api_version: str = "2022-11-28"


@dataclass
class IssueDetails:
    """GitHub issue details"""
    number: int
    title: str
    body: Optional[str]
    state: str
    html_url: str


@dataclass
class PullRequestDetails:
    """GitHub pull request details"""
    number: int
    title: str
    html_url: str
    state: str


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors"""
    pass


class GitHubNotFoundError(GitHubAPIError):
    """Resource not found (404)"""
    pass


class GitHubClient:
    """
    Async GitHub API client.

    Provides methods for interacting with GitHub API for issue implementation workflows.
    Uses a persistent HTTP client for better performance.
    """

    def __init__(self, config: GitHubConfig):
        """
        Initialize GitHub client.

        Args:
            config: GitHub configuration
        """
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

        # Build base headers
        self._headers = {
            "Authorization": f"token {config.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": config.api_version
        }

        # Build repo path
        self._repo_path = f"repos/{config.repo_owner}/{config.repo_name}"

    async def __aenter__(self):
        """Context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()

    async def connect(self):
        """Initialize HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True
            )

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, raising if not connected"""
        if self._client is None:
            raise RuntimeError("GitHubClient not connected. Use async with or call connect()")
        return self._client

    def _build_url(self, path: str) -> str:
        """Build full API URL"""
        return f"{self.config.api_url}/{self._repo_path}/{path}"

    @retry_with_backoff(max_retries=3)
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make authenticated API request.

        Args:
            method: HTTP method
            path: API path (relative to repo)
            **kwargs: Additional request arguments

        Returns:
            Response JSON

        Raises:
            GitHubNotFoundError: Resource not found
            GitHubAPIError: Other API errors
        """
        client = self._get_client()
        url = self._build_url(path)

        try:
            response = await client.request(
                method,
                url,
                headers=self._headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.content else {}

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise GitHubNotFoundError(f"Resource not found: {path}") from e
            raise GitHubAPIError(
                f"GitHub API error ({e.response.status_code}): {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise GitHubAPIError(f"Request failed: {str(e)}") from e

    # ========================================================================
    # Issue Operations
    # ========================================================================

    async def get_issue(self, issue_number: int) -> IssueDetails:
        """
        Get issue details.

        Args:
            issue_number: Issue number

        Returns:
            Issue details
        """
        data = await self._request("GET", f"issues/{issue_number}")

        return IssueDetails(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=data["state"],
            html_url=data["html_url"]
        )

    async def get_issue_comments(self, issue_number: int) -> List[str]:
        """
        Get all comments from an issue.

        Args:
            issue_number: Issue number

        Returns:
            List of comment bodies (excludes '/implement' command)
        """
        data = await self._request("GET", f"issues/{issue_number}/comments")

        # Filter out the /implement command
        return [
            comment["body"]
            for comment in data
            if comment["body"].strip() != "/implement"
        ]

    async def post_issue_comment(self, issue_number: int, body: str) -> None:
        """
        Post a comment on an issue.

        Args:
            issue_number: Issue number
            body: Comment text
        """
        await self._request(
            "POST",
            f"issues/{issue_number}/comments",
            json={"body": body}
        )

    # ========================================================================
    # Repository Operations
    # ========================================================================

    async def get_default_branch(self) -> str:
        """
        Get the repository's default branch name.

        Returns:
            Branch name (e.g., 'main', 'master')
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}"

        response = await client.get(url, headers=self._headers)
        response.raise_for_status()

        return response.json()["default_branch"]

    # ========================================================================
    # File Operations
    # ========================================================================

    async def get_file_content(
        self,
        file_path: str,
        ref: str = "main"
    ) -> Optional[str]:
        """
        Get file content from repository.

        Args:
            file_path: Path to file in repository
            ref: Branch, tag, or commit SHA (default: 'main')

        Returns:
            File content as string, or None if not found
        """
        try:
            data = await self._request(
                "GET",
                f"contents/{file_path}?ref={ref}"
            )

            # Decode base64 content
            if data.get("type") == "file" and data.get("encoding") == "base64":
                content_bytes = base64.b64decode(data["content"])
                return content_bytes.decode("utf-8")

            return None

        except GitHubNotFoundError:
            # File doesn't exist (may be a new file)
            return None

    async def list_directory(
        self,
        dir_path: str = "",
        ref: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        List contents of a directory.

        Args:
            dir_path: Path to directory (empty for root)
            ref: Branch, tag, or commit SHA

        Returns:
            List of directory entries
        """
        path = f"contents/{dir_path}?ref={ref}" if dir_path else f"contents?ref={ref}"
        data = await self._request("GET", path)

        # Ensure we got a list
        if not isinstance(data, list):
            raise GitHubAPIError(f"Path is not a directory: {dir_path}")

        return data

    # ========================================================================
    # Branch Operations
    # ========================================================================

    async def get_branch_sha(self, branch: str) -> str:
        """
        Get SHA of the latest commit on a branch.

        Args:
            branch: Branch name

        Returns:
            Commit SHA
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}/git/ref/heads/{branch}"

        response = await client.get(url, headers=self._headers)
        response.raise_for_status()

        return response.json()["object"]["sha"]

    async def create_branch(self, branch_name: str, base_sha: str) -> None:
        """
        Create a new branch.

        Args:
            branch_name: Name for new branch
            base_sha: SHA to branch from
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}/git/refs"

        payload = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        }

        response = await client.post(url, headers=self._headers, json=payload)
        response.raise_for_status()

    async def update_branch_ref(self, branch_name: str, commit_sha: str) -> None:
        """
        Update branch to point to a commit.

        Args:
            branch_name: Branch name
            commit_sha: Commit SHA
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}/git/refs/heads/{branch_name}"

        payload = {"sha": commit_sha}

        response = await client.patch(url, headers=self._headers, json=payload)
        response.raise_for_status()

    # ========================================================================
    # Git Operations (Low-level)
    # ========================================================================

    async def get_git_tree(
        self,
        tree_sha: str,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Get a Git tree.

        Args:
            tree_sha: Tree SHA or commit SHA
            recursive: If True, fetch tree recursively

        Returns:
            Tree data with file/directory entries
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}/git/trees/{tree_sha}"

        if recursive:
            url += "?recursive=1"

        response = await client.get(url, headers=self._headers)
        response.raise_for_status()

        return response.json()

    async def create_blob(self, content: str) -> str:
        """
        Create a Git blob.

        Args:
            content: File content

        Returns:
            Blob SHA
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}/git/blobs"

        payload = {
            "content": content,
            "encoding": "utf-8"
        }

        response = await client.post(url, headers=self._headers, json=payload)
        response.raise_for_status()

        return response.json()["sha"]

    async def create_tree(
        self,
        base_tree_sha: str,
        file_changes: List[Dict[str, str]]
    ) -> str:
        """
        Create a Git tree.

        Args:
            base_tree_sha: Base tree SHA
            file_changes: List of file changes
                         e.g., [{"path": "file.py", "mode": "100644", "type": "blob", "sha": "..."}]

        Returns:
            Tree SHA
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}/git/trees"

        payload = {
            "base_tree": base_tree_sha,
            "tree": file_changes
        }

        response = await client.post(url, headers=self._headers, json=payload)
        response.raise_for_status()

        return response.json()["sha"]

    async def create_commit(
        self,
        tree_sha: str,
        parent_sha: str,
        message: str
    ) -> str:
        """
        Create a Git commit.

        Args:
            tree_sha: Tree SHA
            parent_sha: Parent commit SHA
            message: Commit message

        Returns:
            Commit SHA
        """
        client = self._get_client()
        url = f"{self.config.api_url}/{self._repo_path}/git/commits"

        payload = {
            "message": message,
            "tree": tree_sha,
            "parents": [parent_sha]
        }

        response = await client.post(url, headers=self._headers, json=payload)
        response.raise_for_status()

        return response.json()["sha"]

    # ========================================================================
    # Pull Request Operations
    # ========================================================================

    async def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str
    ) -> PullRequestDetails:
        """
        Create a pull request.

        Args:
            title: PR title
            body: PR description
            head_branch: Source branch
            base_branch: Target branch

        Returns:
            Pull request details
        """
        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch
        }

        data = await self._request("POST", "pulls", json=payload)

        return PullRequestDetails(
            number=data["number"],
            title=data["title"],
            html_url=data["html_url"],
            state=data["state"]
        )
