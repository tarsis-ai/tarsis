"""
Commit message generation and validation module.

Provides conventional commits support with AI-powered message generation,
validation, and formatting utilities.

**Key Features:**
- Conventional Commits specification compliance
- AI-powered commit message generation
- Comprehensive message validation
- Support for multiple commit types and scopes

**Main Components:**
- **conventional**: Conventional commits format and parsing
- **validator**: Message validation and linting
- **message_generator**: AI-powered message generation
"""

from .conventional import (
    CommitType,
    ConventionalCommit,
    detect_commit_type_from_files,
    detect_commit_type_from_content,
    detect_scope_from_files,
    parse_conventional_commit,
    format_conventional_commit,
    is_valid_description
)

from .validator import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    validate_commit_message,
    suggest_improvements,
    is_conventional_commit,
    get_commit_type
)

from .message_generator import (
    FileChange,
    CommitContext,
    GeneratedCommitMessage,
    generate_commit_message,
    generate_with_retry,
    extract_changes_from_git_status
)

__all__ = [
    # Conventional commits
    "CommitType",
    "ConventionalCommit",
    "detect_commit_type_from_files",
    "detect_commit_type_from_content",
    "detect_scope_from_files",
    "parse_conventional_commit",
    "format_conventional_commit",
    "is_valid_description",
    # Validation
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "validate_commit_message",
    "suggest_improvements",
    "is_conventional_commit",
    "get_commit_type",
    # Message generation
    "FileChange",
    "CommitContext",
    "GeneratedCommitMessage",
    "generate_commit_message",
    "generate_with_retry",
    "extract_changes_from_git_status",
]
