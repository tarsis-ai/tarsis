"""
Retry utility with exponential backoff for handling transient failures.

Provides decorator and configuration for automatic retry logic.
"""

import os
import asyncio
import logging
import random
from functools import wraps
from typing import Callable, TypeVar, Optional, Tuple, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


# Exceptions that should be retried
RETRYABLE_EXCEPTIONS = (
    # Network errors
    ConnectionError,
    TimeoutError,
    # HTTP client errors (from httpx)
    Exception,  # We'll check the message for specific errors
)

# HTTP status codes that should be retried
RETRYABLE_STATUS_CODES = {
    429,  # Too Many Requests (rate limit)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# HTTP status codes that should NOT be retried
NON_RETRYABLE_STATUS_CODES = {
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
    422,  # Unprocessable Entity
}


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error should be retried
    """
    error_str = str(error).lower()

    # Check for HTTP status codes in error message
    for code in NON_RETRYABLE_STATUS_CODES:
        if f"status code {code}" in error_str or f"{code}" in error_str:
            return False

    for code in RETRYABLE_STATUS_CODES:
        if f"status code {code}" in error_str or f"{code}" in error_str:
            return True

    # Check error type
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    # Check for network-related error messages
    network_keywords = [
        "connection",
        "timeout",
        "network",
        "unreachable",
        "unavailable",
        "temporarily",
    ]

    return any(keyword in error_str for keyword in network_keywords)


def calculate_delay(
    attempt: int,
    base_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter: bool
) -> float:
    """
    Calculate delay for retry attempt with exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    # Calculate exponential delay
    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    # Add jitter to prevent thundering herd
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)

    return delay


def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exponential_base: Optional[float] = None,
    jitter: bool = True
):
    """
    Decorator for retrying async functions with exponential backoff.

    Reads configuration from environment variables if not provided:
    - MAX_RETRIES: Maximum number of retry attempts (default: 3)
    - RETRY_BACKOFF_BASE: Exponential base for backoff (default: 2.0)
    - RETRY_MAX_DELAY: Maximum delay between retries (default: 60.0)

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def my_function():
            # Function that may fail transiently
            pass

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays

    Returns:
        Decorated function with retry logic
    """
    # Read from environment if not provided
    if max_retries is None:
        max_retries = int(os.getenv("MAX_RETRIES", "3"))
    if base_delay is None:
        base_delay = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
    if max_delay is None:
        max_delay = float(os.getenv("RETRY_MAX_DELAY", "60.0"))
    if exponential_base is None:
        exponential_base = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Try to execute the function
                    result = await func(*args, **kwargs)

                    # Log success if this was a retry
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if attempt >= max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
                        raise

                    if not is_retryable_error(e):
                        logger.debug(f"{func.__name__} failed with non-retryable error: {e}")
                        raise

                    # Calculate delay
                    delay = calculate_delay(
                        attempt,
                        base_delay,
                        exponential_base,
                        max_delay,
                        jitter
                    )

                    logger.warning(
                        f"{func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}, "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    # Wait before retrying
                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator
