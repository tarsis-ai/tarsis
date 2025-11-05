# GitHub API Client

Modern, type-safe GitHub API client for Tarsis.

## Overview

The `github` module provides a clean, async interface to GitHub API operations needed for issue implementation workflows.

## Key Features

- **Type Safety**: Dataclasses for all request/response structures
- **Async-First**: All operations use async/await
- **Resource Management**: Persistent HTTP client with proper lifecycle
- **Error Handling**: Custom exceptions for different error types
- **Context Manager**: Support for `async with` pattern

## Usage

### Basic Usage

```python
from tarsis.github import GitHubClient, GitHubConfig

# Create configuration
config = GitHubConfig(
    token="ghp_xxx",
    repo_owner="myorg",
    repo_name="myrepo"
)

# Use as context manager (recommended)
async with GitHubClient(config) as client:
    # Get issue details
    issue = await client.get_issue(123)
    print(f"Issue: {issue.title}")

    # Get comments
    comments = await client.get_issue_comments(123)

    # Post a comment
    await client.post_issue_comment(123, "Working on this!")
```

### Manual Lifecycle Management

```python
client = GitHubClient(config)
await client.connect()

try:
    issue = await client.get_issue(123)
finally:
    await client.close()
```

## API Reference

### Configuration

**GitHubConfig**
- `token`: GitHub API token
- `repo_owner`: Repository owner
- `repo_name`: Repository name
- `api_url`: API base URL (default: "https://api.github.com")
- `api_version`: API version (default: "2022-11-28")

### Issue Operations

- `get_issue(issue_number)` → `IssueDetails`
- `get_issue_comments(issue_number)` → `List[str]`
- `post_issue_comment(issue_number, body)` → `None`

### File Operations

- `get_file_content(file_path, ref="main")` → `Optional[str]`
- `list_directory(dir_path="", ref="main")` → `List[Dict]`

### Branch Operations

- `get_default_branch()` → `str`
- `get_branch_sha(branch)` → `str`
- `create_branch(branch_name, base_sha)` → `None`
- `update_branch_ref(branch_name, commit_sha)` → `None`

### Git Operations (Low-level)

- `create_blob(content)` → `str`  (returns blob SHA)
- `create_tree(base_tree_sha, file_changes)` → `str`  (returns tree SHA)
- `create_commit(tree_sha, parent_sha, message)` → `str`  (returns commit SHA)

### Pull Request Operations

- `create_pull_request(title, body, head_branch, base_branch)` → `PullRequestDetails`

## Data Classes

### IssueDetails
- `number`: Issue number
- `title`: Issue title
- `body`: Issue description
- `state`: Issue state (open/closed)
- `html_url`: Issue URL

### PullRequestDetails
- `number`: PR number
- `title`: PR title
- `html_url`: PR URL
- `state`: PR state

## Error Handling

```python
from tarsis.github import GitHubClient, GitHubAPIError, GitHubNotFoundError

async with GitHubClient(config) as client:
    try:
        content = await client.get_file_content("nonexistent.py")
        # Returns None for 404

    except GitHubAPIError as e:
        # Handle other API errors
        print(f"API error: {e}")
```

## Usage in Tools

Tools should use the new GitHubClient directly:

```python
from tarsis.github import GitHubClient, GitHubConfig
import os

# Create client
config = GitHubConfig(
    token=os.getenv("GITHUB_TOKEN"),
    repo_owner=os.getenv("GITHUB_REPO_OWNER"),
    repo_name=os.getenv("GITHUB_REPO_NAME")
)

async with GitHubClient(config) as client:
    issue = await client.get_issue(123)
    print(f"Working on: {issue.title}")
```

## Architecture Notes

- **Single HTTP Client**: Reuses the same `httpx.AsyncClient` across requests for better performance
- **Lazy Connection**: Client connects only when first request is made (or when `connect()` is called)
- **Automatic Cleanup**: Context manager ensures client is properly closed
- **Type Safety**: All responses use typed dataclasses instead of raw dicts
