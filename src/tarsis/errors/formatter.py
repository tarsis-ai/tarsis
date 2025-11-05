"""
Error message formatting for user-friendly GitHub comments.
"""

import logging
from typing import Optional
from .categories import ErrorCategory, categorize_error

logger = logging.getLogger(__name__)


class ErrorFormatter:
    """
    Formats errors into user-friendly messages for GitHub comments.
    """

    # Suggestions for each error category
    SUGGESTIONS = {
        ErrorCategory.CONFIGURATION: [
            "Check your `.env` file for missing or incorrect values",
            "Verify your API keys are valid and not expired",
            "Ensure all required environment variables are set"
        ],
        ErrorCategory.AUTH: [
            "Verify your GitHub token has the correct permissions",
            "Check that your LLM provider API key is valid",
            "Ensure your API keys haven't expired"
        ],
        ErrorCategory.TIMEOUT: [
            "Try increasing the timeout settings in your `.env` file",
            "For Ollama: Set `OLLAMA_TIMEOUT=0` for unlimited timeout",
            "Check if the model or API service is responding"
        ],
        ErrorCategory.NETWORK: [
            "Check your internet connection",
            "Verify the API endpoint is accessible",
            "Try again in a few moments"
        ],
        ErrorCategory.API: [
            "Check the service status page for outages",
            "Try again in a few moments",
            "Verify your request parameters are valid"
        ],
        ErrorCategory.TOOL: [
            "Check the tool input parameters",
            "Verify the tool has necessary permissions",
            "Review the error details for specific issues"
        ],
        ErrorCategory.VALIDATION: [
            "Review the validation errors and fix the code",
            "Run tests locally to debug the issue",
            "Check syntax and type errors"
        ],
        ErrorCategory.INTERNAL: [
            "Try running the task again",
            "Check the server logs for more details",
            "Report this issue if it persists"
        ]
    }

    # Emojis for each category
    EMOJIS = {
        ErrorCategory.CONFIGURATION: "⚙️",
        ErrorCategory.AUTH: "🔒",
        ErrorCategory.TIMEOUT: "⏱️",
        ErrorCategory.NETWORK: "🌐",
        ErrorCategory.API: "🔌",
        ErrorCategory.TOOL: "🔧",
        ErrorCategory.VALIDATION: "✅",
        ErrorCategory.INTERNAL: "⚠️"
    }

    @staticmethod
    def format_error_for_github(
        error: Exception,
        issue_number: int,
        include_traceback: bool = False
    ) -> str:
        """
        Format an error as a GitHub comment.

        Args:
            error: The exception to format
            issue_number: The issue number this error relates to
            include_traceback: Whether to include technical details (default: False)

        Returns:
            Formatted markdown string for GitHub comment
        """
        # Categorize the error
        category, explanation = categorize_error(error)
        emoji = ErrorFormatter.EMOJIS.get(category, "❌")
        suggestions = ErrorFormatter.SUGGESTIONS.get(category, [])

        # Build the message
        lines = [
            f"{emoji} **Task Failed - {category.value.replace('_', ' ').title()} Error**",
            "",
            f"Issue #{issue_number} could not be completed due to an error.",
            "",
            "### What Happened",
            f"{explanation}: {str(error)[:200]}",
            ""
        ]

        # Add suggestions
        if suggestions:
            lines.append("### 💡 Suggestions")
            for suggestion in suggestions:
                lines.append(f"- {suggestion}")
            lines.append("")

        # Add technical details if requested
        if include_traceback:
            lines.append("<details>")
            lines.append("<summary>Technical Details (click to expand)</summary>")
            lines.append("")
            lines.append("```")
            lines.append(f"Error Type: {type(error).__name__}")
            lines.append(f"Error Message: {str(error)}")
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        # Add footer
        lines.append("---")
        lines.append("*This is an automated message from Tarsis*")

        return "\n".join(lines)

    @staticmethod
    def format_error_concise(error: Exception) -> str:
        """
        Format an error concisely for logs or inline display.

        Args:
            error: The exception to format

        Returns:
            Concise error string
        """
        category, explanation = categorize_error(error)
        emoji = ErrorFormatter.EMOJIS.get(category, "❌")

        return f"{emoji} {category.value.upper()}: {explanation} - {str(error)[:100]}"


def format_error_for_user(
    error: Exception,
    issue_number: Optional[int] = None,
    format_type: str = "github"
) -> str:
    """
    Convenience function to format an error for display to users.

    Args:
        error: The exception to format
        issue_number: Optional issue number for GitHub format
        format_type: Format type ("github" or "concise")

    Returns:
        Formatted error message
    """
    if format_type == "github" and issue_number:
        return ErrorFormatter.format_error_for_github(error, issue_number)
    else:
        return ErrorFormatter.format_error_concise(error)
