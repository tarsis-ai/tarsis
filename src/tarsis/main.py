"""
Main entry point for Tarsis - AI-powered GitHub issue implementation assistant.

Uses a recursive agent loop with tool calling for autonomous issue implementation.
"""

import uvicorn
import asyncio
import os
import logging
from fastapi import FastAPI, Request, HTTPException
import json

# Import agentic components
from .agent import AgentTask, TaskConfig
from .llm import create_llm_provider
from .tools import create_default_tool_executor
from .github import GitHubClient, GitHubConfig
from .errors import ErrorFormatter

logger = logging.getLogger(__name__)


app = FastAPI(title="Tarsis - AI GitHub Issue Assistant")

# Global GitHub client
_github_client: GitHubClient = None


def get_github_client() -> GitHubClient:
    """Get or create GitHub client instance."""
    global _github_client

    if _github_client is None:
        config = GitHubConfig(
            token=os.getenv("GITHUB_TOKEN"),
            repo_owner=os.getenv("GITHUB_REPO_OWNER"),
            repo_name=os.getenv("GITHUB_REPO_NAME")
        )
        _github_client = GitHubClient(config)

    return _github_client


async def process_issue_with_agent(payload: dict) -> None:
    """
    Process a GitHub issue using the agentic architecture.

    Args:
        payload: GitHub webhook payload
    """
    issue_number = payload["issue"]["number"]
    repo_owner = os.getenv("GITHUB_REPO_OWNER")
    repo_name = os.getenv("GITHUB_REPO_NAME")

    # Get GitHub client
    github = get_github_client()
    await github.connect()

    try:
        # Post initial comment
        await github.post_issue_comment(issue_number, "🤖 Sure! Starting work on the implementation...")

        # Get default branch
        default_branch = await github.get_default_branch()

        # Create task configuration
        task_config = TaskConfig(
            issue_number=issue_number,
            repo_owner=repo_owner,
            repo_name=repo_name,
            default_branch=default_branch,
            max_iterations=25,
            max_consecutive_mistakes=3,
            auto_approve=False  # Could be configured per repo
        )

        # Create LLM provider
        # Get provider from env or default to Ollama
        provider_type = os.getenv("LLM_PROVIDER", "ollama")
        model_id = os.getenv("LLM_MODEL_ID")  # Optional, uses provider defaults
        api_key = os.getenv("LLM_API_KEY")  # For Anthropic/OpenAI

        llm_provider = create_llm_provider(
            provider_type=provider_type,
            model_id=model_id,
            api_key=api_key
        )

        # Create tool executor with all tools
        tool_executor = create_default_tool_executor()

        # Create and execute agent task
        agent = AgentTask(
            config=task_config,
            llm_provider=llm_provider,
            tool_executor=tool_executor
        )

        # Build initial prompt from issue data
        issue = await github.get_issue(issue_number)
        comments = await github.get_issue_comments(issue_number)

        initial_prompt = f"""Please implement the following GitHub issue:

# Issue #{issue_number}: {issue.title}

## Description
{issue.body}
"""

        if comments:
            initial_prompt += "\n## Comments\n"
            for i, comment in enumerate(comments, 1):
                initial_prompt += f"\n### Comment {i}\n{comment}\n"

        initial_prompt += """

Please:
1. Read the issue carefully to understand requirements
2. Create an implementation plan
3. Identify and read relevant files
4. Implement the necessary changes
5. Create a pull request with your changes
6. Use attempt_completion when done

Start by using the read_issue tool to confirm your understanding."""

        # Execute the agent task
        result = await agent.execute(initial_prompt)

        # Post completion comment
        completion_msg = result.get("completion_message", "Task completed")
        pr_url = result.get("pr_url")

        final_comment = f"""✅ **Implementation completed!**

{completion_msg}

**Stats:**
- Iterations: {result['iterations']}
- Files modified: {len(result['files_modified'])}
"""

        if pr_url:
            final_comment += f"\n🔗 **Pull Request**: {pr_url}"

        if result['files_modified']:
            final_comment += "\n\n**Modified files:**\n"
            for file in result['files_modified']:
                final_comment += f"- `{file}`\n"

        await github.post_issue_comment(issue_number, final_comment)

        logger.info(f"✅ Successfully completed issue #{issue_number}")

    except Exception as e:
        logger.error(f"Error processing issue #{issue_number}: {str(e)}", exc_info=True)

        # Determine if we should include technical details
        # Default to False for user-facing messages, but can be enabled via env var
        include_traceback = os.getenv("ERROR_INCLUDE_TRACEBACK", "false").lower() in ("true", "1", "yes")

        # Format error for user-friendly GitHub comment
        error_comment = ErrorFormatter.format_error_for_github(
            error=e,
            issue_number=issue_number,
            include_traceback=include_traceback
        )

        # Post formatted error comment
        await github.post_issue_comment(issue_number, error_comment)

        logger.info(f"Posted error comment to issue #{issue_number}")

    finally:
        await github.close()


@app.post("/webhook")
async def github_webhook(request: Request):
    """
    GitHub webhook endpoint.

    Listens for issue comment events with "/implement" command.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    # TODO: Validate webhook signature for production

    # Check if this is an issue comment created event
    if "issue" not in payload or "comment" not in payload or payload.get("action") != "created":
        return {"status": "ignored", "reason": "not an issue comment created event"}

    # Check for /implement command
    comment_body = payload["comment"]["body"]
    if comment_body.strip() == "/implement":
        logger.info(f"/implement command received on issue: {payload['issue']['number']}")

        # Process in background
        asyncio.create_task(process_issue_with_agent(payload))

        return {"status": "processing started"}

    return {"status": "ignored"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tarsis",
        "architecture": "agentic"
    }


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "name": "Tarsis",
        "description": "AI-Powered GitHub Issue Implementation Assistant",
        "version": "v0.1.0",
        "architecture": "Recursive agent loop with tool calling",
        "features": [
            "Autonomous issue implementation",
            "Multi-step planning and execution",
            "Tool-based GitHub integration",
            "Local LLM support (Ollama)",
            "Modular prompt system"
        ]
    }


if __name__ == "__main__":
    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║                    Tarsis Agent v0.1                         ║
║          AI-Powered GitHub Issue Implementation              ║
╚══════════════════════════════════════════════════════════════╝

Architecture: Custom agentic loop with tool calling
Pattern: Recursive agent with multi-provider LLM support

Starting server...
    """)

    uvicorn.run(app, host="127.0.0.1", port=8000)
