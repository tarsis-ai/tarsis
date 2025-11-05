"""
Error categorization for user-friendly error messages.
"""

from enum import Enum
from typing import Tuple


class ErrorCategory(Enum):
    """Categories of errors that can occur in Tarsis"""
    CONFIGURATION = "configuration"
    API = "api"
    TIMEOUT = "timeout"
    TOOL = "tool"
    VALIDATION = "validation"
    INTERNAL = "internal"
    NETWORK = "network"
    AUTH = "authentication"


def categorize_error(error: Exception) -> Tuple[ErrorCategory, str]:
    """
    Categorize an error and provide a user-friendly explanation.

    Args:
        error: The exception to categorize

    Returns:
        Tuple of (ErrorCategory, explanation)
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Configuration errors
    if "api key" in error_str or "token" in error_str or "config" in error_str:
        return (
            ErrorCategory.CONFIGURATION,
            "Configuration error - please check your API keys and environment variables"
        )

    if "env" in error_str or "environment" in error_str:
        return (
            ErrorCategory.CONFIGURATION,
            "Missing or invalid environment variable"
        )

    # Authentication errors
    if "401" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
        return (
            ErrorCategory.AUTH,
            "Authentication failed - please check your API keys"
        )

    # Timeout errors
    if "timeout" in error_str or error_type == "TimeoutError":
        return (
            ErrorCategory.TIMEOUT,
            "Operation timed out - the model or API took too long to respond"
        )

    # Network errors
    if any(keyword in error_str for keyword in ["connection", "network", "unreachable"]):
        return (
            ErrorCategory.NETWORK,
            "Network error - please check your internet connection"
        )

    # Rate limit errors
    if "429" in error_str or "rate limit" in error_str:
        return (
            ErrorCategory.API,
            "Rate limit exceeded - too many requests"
        )

    # API errors
    if any(code in error_str for code in ["400", "404", "500", "502", "503"]):
        return (
            ErrorCategory.API,
            "API error - the service returned an error"
        )

    # Tool errors
    if "tool" in error_str:
        return (
            ErrorCategory.TOOL,
            "Tool execution failed"
        )

    # Validation errors
    if any(keyword in error_str for keyword in ["test", "lint", "syntax", "validation"]):
        return (
            ErrorCategory.VALIDATION,
            "Validation failed - code did not pass checks"
        )

    # Iteration limit
    if "iteration" in error_str or "maximum" in error_str:
        return (
            ErrorCategory.INTERNAL,
            "Task aborted - reached iteration limit without completing"
        )

    # Default to internal error
    return (
        ErrorCategory.INTERNAL,
        "An unexpected error occurred"
    )
