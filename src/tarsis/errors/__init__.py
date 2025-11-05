"""
Error handling and formatting for Tarsis.
"""

from .formatter import ErrorFormatter, format_error_for_user
from .categories import ErrorCategory, categorize_error

__all__ = ["ErrorFormatter", "format_error_for_user", "ErrorCategory", "categorize_error"]
