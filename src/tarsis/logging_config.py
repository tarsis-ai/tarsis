"""
Logging configuration for Tarsis.

Configures structured logging based on environment variables:
- LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL
- LOG_FORMAT: simple, detailed, json
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON for easy parsing and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


def configure_logging(
    level: Optional[str] = None,
    format_style: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to LOG_LEVEL env var or INFO.
        format_style: Format style (simple, detailed, json).
                     Defaults to LOG_FORMAT env var or simple.
    """
    # Get configuration from environment or parameters
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_format = (format_style or os.getenv("LOG_FORMAT", "simple")).lower()

    # Validate log level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        sys.stderr.write(f"Warning: Invalid LOG_LEVEL '{log_level}', defaulting to INFO\n")
        log_level = "INFO"

    # Convert to logging constant
    numeric_level = getattr(logging, log_level)

    # Create formatter based on format style
    if log_format == "json":
        formatter = JSONFormatter()
    elif log_format == "detailed":
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:  # simple or default
        formatter = logging.Formatter(
            fmt="%(levelname)s - %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    # Log configuration
    root_logger.info(f"Logging configured: level={log_level}, format={log_format}")

    # Set log levels for noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
