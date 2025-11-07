"""
AI-powered commit message generation.

Uses LLM providers to generate high-quality conventional commit messages
based on file changes and context.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..llm.provider import ILLMProvider
from .conventional import (
    CommitType,
    detect_commit_type_from_files,
    detect_commit_type_from_content,
    detect_scope_from_files,
    format_conventional_commit
)
from .validator import validate_commit_message, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """Represents a file change for commit message generation."""

    path: str
    change_type: str  # "create", "update", "delete", "rename"
    additions: int = 0
    deletions: int = 0
    diff_snippet: Optional[str] = None  # First few lines of diff
    old_path: Optional[str] = None  # For renames


@dataclass
class CommitContext:
    """Context information for generating commit message."""

    file_changes: List[FileChange]
    branch_name: Optional[str] = None
    issue_number: Optional[str] = None
    issue_title: Optional[str] = None
    additional_context: Optional[str] = None


@dataclass
class GeneratedCommitMessage:
    """Result of AI commit message generation."""

    message: str
    commit_type: CommitType
    scope: Optional[str]
    confidence: float  # 0.0-1.0, how confident the generator is
    reasoning: Optional[str] = None  # Why this message was chosen
    alternatives: Optional[List[str]] = None  # Alternative messages


# System prompt for commit message generation
COMMIT_MESSAGE_SYSTEM_PROMPT = """You are an expert at writing clear, concise commit messages following the Conventional Commits specification.

Your task is to generate a commit message based on file changes provided by the user.

**Conventional Commits Format:**
```
type(scope): description

[optional body]

[optional footer]
```

**Standard Types:**
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code formatting (no logic change)
- refactor: Code restructuring (no behavior change)
- test: Adding or updating tests
- chore: Maintenance, dependencies
- perf: Performance improvements
- build: Build system or dependencies
- ci: CI/CD configuration
- revert: Reverting previous commits

**Best Practices:**
1. Use imperative mood (e.g., "add" not "added" or "adds")
2. Start description with lowercase
3. No period at end of description
4. Keep header under 72 characters
5. Scope should indicate affected area (file/module/component)
6. Body explains WHY, not WHAT (code shows what)
7. Reference issues in footer if applicable

**Output Format:**
Respond with ONLY the commit message, no additional explanation."""


def _build_change_summary(changes: List[FileChange]) -> str:
    """Build a concise summary of file changes for the LLM."""
    lines = [f"**Total changes:** {len(changes)} file(s) modified\n"]

    # Group by change type
    creates = [c for c in changes if c.change_type == "create"]
    updates = [c for c in changes if c.change_type == "update"]
    deletes = [c for c in changes if c.change_type == "delete"]
    renames = [c for c in changes if c.change_type == "rename"]

    if creates:
        lines.append(f"\n**Created ({len(creates)}):**")
        for change in creates[:5]:  # Limit to 5 examples
            lines.append(f"- {change.path} (+{change.additions} lines)")
            if change.diff_snippet:
                lines.append(f"  ```\n{change.diff_snippet}\n  ```")

    if updates:
        lines.append(f"\n**Updated ({len(updates)}):**")
        for change in updates[:5]:
            lines.append(f"- {change.path} (+{change.additions}/-{change.deletions})")
            if change.diff_snippet:
                lines.append(f"  ```\n{change.diff_snippet}\n  ```")

    if renames:
        lines.append(f"\n**Renamed ({len(renames)}):**")
        for change in renames[:5]:
            lines.append(f"- {change.old_path} → {change.path}")

    if deletes:
        lines.append(f"\n**Deleted ({len(deletes)}):**")
        for change in deletes[:5]:
            lines.append(f"- {change.path} (-{change.deletions} lines)")

    # Add note if truncated
    total_shown = min(len(creates), 5) + min(len(updates), 5) + min(len(renames), 5) + min(len(deletes), 5)
    if len(changes) > total_shown:
        lines.append(f"\n... and {len(changes) - total_shown} more file(s)")

    return "\n".join(lines)


def _build_generation_prompt(context: CommitContext) -> str:
    """Build user prompt for commit message generation."""
    parts = ["Generate a conventional commit message for the following changes:\n"]

    # Add change summary
    parts.append(_build_change_summary(context.file_changes))

    # Add additional context if available
    if context.issue_number or context.issue_title:
        parts.append("\n**Related Issue:**")
        if context.issue_number:
            parts.append(f"- Number: #{context.issue_number}")
        if context.issue_title:
            parts.append(f"- Title: {context.issue_title}")

    if context.branch_name:
        parts.append(f"\n**Branch:** {context.branch_name}")

    if context.additional_context:
        parts.append(f"\n**Additional Context:**\n{context.additional_context}")

    parts.append("\n**Instructions:**")
    parts.append("1. Analyze the changes and determine the appropriate commit type")
    parts.append("2. Identify the scope (affected component/module)")
    parts.append("3. Write a concise description in imperative mood")
    parts.append("4. Add body if the change is complex or needs explanation")
    parts.append("5. Reference the issue in footer if applicable")
    parts.append("\nGenerate the commit message now:")

    return "\n".join(parts)


async def generate_commit_message(
    context: CommitContext,
    llm_provider: ILLMProvider,
    temperature: float = 0.3,
    max_attempts: int = 3
) -> GeneratedCommitMessage:
    """
    Generate a commit message using AI.

    Args:
        context: Context about the changes to commit
        llm_provider: LLM provider to use for generation
        temperature: Sampling temperature (lower = more focused)
        max_attempts: Maximum validation retry attempts

    Returns:
        Generated commit message with metadata

    Raises:
        ValueError: If context is invalid
        RuntimeError: If generation fails after max_attempts
    """
    if not context.file_changes:
        raise ValueError("No file changes provided for commit message generation")

    # Build prompt
    user_prompt = _build_generation_prompt(context)

    logger.info(f"Generating commit message for {len(context.file_changes)} file(s)")

    # Try generating and validating
    best_message = None
    best_score = -1

    for attempt in range(max_attempts):
        try:
            # Call LLM
            response = await llm_provider.create_message(
                system_prompt=COMMIT_MESSAGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                tools=None,  # No tool calling needed
                temperature=temperature,
                max_tokens=500  # Commit messages are short
            )

            # Extract message from response
            if isinstance(response.content, str):
                generated_message = response.content.strip()
            elif isinstance(response.content, list):
                # Handle content blocks (take first text block)
                text_blocks = [
                    block.get("text", "")
                    for block in response.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                generated_message = text_blocks[0].strip() if text_blocks else ""
            else:
                logger.warning(f"Unexpected response content type: {type(response.content)}")
                continue

            if not generated_message:
                logger.warning("Empty message generated, retrying...")
                continue

            # Validate the generated message
            validation = validate_commit_message(generated_message, strict=False)

            # Calculate score (lower is better)
            error_count = len(validation.errors)
            warning_count = len(validation.warnings)
            score = -(error_count * 10 + warning_count)  # Negative so higher is better

            # Keep track of best message
            if score > best_score:
                best_score = score
                best_message = generated_message

            # If valid, we're done
            if validation.valid:
                logger.info(f"Generated valid commit message on attempt {attempt + 1}")

                # Extract metadata
                parsed = validation.parsed_commit
                if parsed:
                    return GeneratedCommitMessage(
                        message=generated_message,
                        commit_type=parsed.type,
                        scope=parsed.scope,
                        confidence=1.0 - (warning_count * 0.1),  # Reduce confidence for warnings
                        reasoning=None
                    )

            # If not valid but this is our last attempt, use best message
            if attempt == max_attempts - 1:
                logger.warning(
                    f"Failed to generate valid message after {max_attempts} attempts. "
                    f"Using best attempt with score {best_score}"
                )
                break

        except Exception as e:
            logger.error(f"Error generating commit message (attempt {attempt + 1}): {e}")
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Failed to generate commit message after {max_attempts} attempts") from e

    # Fallback: Use heuristic generation if AI failed
    if not best_message:
        logger.warning("AI generation failed, falling back to rule-based generation")
        return _generate_heuristic_message(context)

    # Parse best message for metadata
    from .conventional import parse_conventional_commit
    parsed = parse_conventional_commit(best_message)

    return GeneratedCommitMessage(
        message=best_message,
        commit_type=parsed.type if parsed else CommitType.CHORE,
        scope=parsed.scope if parsed else None,
        confidence=0.5,  # Lower confidence since it has issues
        reasoning="Generated message has validation warnings"
    )


def _generate_heuristic_message(context: CommitContext) -> GeneratedCommitMessage:
    """
    Generate commit message using rule-based heuristics (fallback).

    Used when AI generation fails.
    """
    file_paths = [c.path for c in context.file_changes]

    # Detect type from files
    commit_type = detect_commit_type_from_files(file_paths)

    # If still no type, use change types
    if not commit_type:
        creates = sum(1 for c in context.file_changes if c.change_type == "create")
        deletes = sum(1 for c in context.file_changes if c.change_type == "delete")

        if creates > 0 and deletes == 0:
            commit_type = CommitType.FEAT
        elif deletes > creates:
            commit_type = CommitType.CHORE
        else:
            commit_type = CommitType.REFACTOR

    # Detect scope
    scope = detect_scope_from_files(file_paths)

    # Build description
    if len(context.file_changes) == 1:
        change = context.file_changes[0]
        if change.change_type == "create":
            description = f"add {change.path}"
        elif change.change_type == "delete":
            description = f"remove {change.path}"
        elif change.change_type == "rename":
            description = f"rename {change.old_path} to {change.path}"
        else:
            description = f"update {change.path}"
    else:
        description = f"update {len(context.file_changes)} files"
        if scope:
            description = f"update {scope} module"

    # Build footer
    footer = None
    if context.issue_number:
        footer = f"Closes #{context.issue_number}"

    # Format message
    message = format_conventional_commit(
        commit_type=commit_type,
        description=description,
        scope=scope,
        footer=footer
    )

    return GeneratedCommitMessage(
        message=message,
        commit_type=commit_type,
        scope=scope,
        confidence=0.3,  # Low confidence for heuristic
        reasoning="Generated using rule-based heuristics (AI generation failed)"
    )


async def generate_with_retry(
    context: CommitContext,
    llm_provider: ILLMProvider,
    user_feedback: Optional[str] = None,
    previous_attempt: Optional[str] = None
) -> GeneratedCommitMessage:
    """
    Generate commit message with user feedback from previous attempt.

    Useful for interactive refinement of commit messages.

    Args:
        context: Commit context
        llm_provider: LLM provider
        user_feedback: User's feedback on previous attempt
        previous_attempt: Previously generated message

    Returns:
        Refined commit message
    """
    # Build enhanced prompt with feedback
    base_prompt = _build_generation_prompt(context)

    if previous_attempt and user_feedback:
        enhanced_prompt = f"""{base_prompt}

**Previous Attempt:**
```
{previous_attempt}
```

**User Feedback:**
{user_feedback}

**Instructions:**
Please generate an improved commit message addressing the user's feedback."""

        messages = [{"role": "user", "content": enhanced_prompt}]
    else:
        messages = [{"role": "user", "content": base_prompt}]

    return await generate_commit_message(context, llm_provider)


def extract_changes_from_git_status(
    status_output: str,
    diff_output: Optional[str] = None
) -> List[FileChange]:
    """
    Extract file changes from git status output.

    Helper function to parse git status and create FileChange objects.

    Args:
        status_output: Output from 'git status --porcelain'
        diff_output: Optional output from 'git diff --stat'

    Returns:
        List of FileChange objects
    """
    changes = []

    # Parse git status --porcelain output
    # Format: XY PATH or XY PATH -> NEW_PATH
    for line in status_output.strip().split("\n"):
        if not line:
            continue

        status_code = line[:2]
        path_part = line[3:]

        # Detect change type
        if status_code in ("A ", "??"):
            change_type = "create"
            path = path_part
            old_path = None
        elif status_code in ("D ", " D"):
            change_type = "delete"
            path = path_part
            old_path = None
        elif "R" in status_code:
            change_type = "rename"
            parts = path_part.split(" -> ")
            old_path = parts[0] if len(parts) > 1 else None
            path = parts[1] if len(parts) > 1 else path_part
        else:
            change_type = "update"
            path = path_part
            old_path = None

        changes.append(FileChange(
            path=path,
            change_type=change_type,
            old_path=old_path
        ))

    return changes
