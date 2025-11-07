"""
Commit message validation for conventional commits.

Provides comprehensive validation of commit messages to ensure they follow
conventional commits specification and best practices.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .conventional import (
    CommitType,
    ConventionalCommit,
    parse_conventional_commit,
    is_valid_description
)


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    ERROR = "error"      # Must fix (blocks commit)
    WARNING = "warning"  # Should fix (best practice)
    INFO = "info"        # Nice to have (suggestion)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in commit message."""

    severity: ValidationSeverity
    message: str
    line: Optional[int] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Format issue for display."""
        prefix = {
            ValidationSeverity.ERROR: "❌ ERROR",
            ValidationSeverity.WARNING: "⚠️  WARNING",
            ValidationSeverity.INFO: "ℹ️  INFO"
        }[self.severity]

        msg = f"{prefix}: {self.message}"
        if self.line is not None:
            msg += f" (line {self.line})"
        if self.suggestion:
            msg += f"\n   Suggestion: {self.suggestion}"

        return msg


@dataclass
class ValidationResult:
    """Result of commit message validation."""

    valid: bool
    issues: List[ValidationIssue]
    parsed_commit: Optional[ConventionalCommit] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> List[ValidationIssue]:
        """Get only info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def format_report(self) -> str:
        """Format validation report for display."""
        if self.valid and not self.issues:
            return "✅ Commit message is valid!"

        lines = []
        if self.valid:
            lines.append("✅ Commit message is valid (with suggestions):")
        else:
            lines.append("❌ Commit message validation failed:")

        lines.append("")

        # Group by severity
        if self.errors:
            lines.append("Errors:")
            for issue in self.errors:
                lines.append(f"  {issue}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for issue in self.warnings:
                lines.append(f"  {issue}")
            lines.append("")

        if self.infos:
            lines.append("Suggestions:")
            for issue in self.infos:
                lines.append(f"  {issue}")

        return "\n".join(lines)


# Imperative mood verbs (should start description)
IMPERATIVE_VERBS = {
    "add", "remove", "fix", "update", "implement", "refactor", "improve",
    "change", "rename", "move", "delete", "create", "introduce", "extract",
    "optimize", "enhance", "simplify", "clean", "merge", "split", "convert",
    "upgrade", "downgrade", "deprecate", "replace", "revert", "bump", "release"
}

# Non-imperative patterns to detect (common mistakes)
NON_IMPERATIVE_PATTERNS = [
    r"^(adds|added|adding)\b",
    r"^(removes|removed|removing)\b",
    r"^(fixes|fixed|fixing)\b",
    r"^(updates|updated|updating)\b",
    r"^(implements|implemented|implementing)\b",
    r"^(refactors|refactored|refactoring)\b",
    r"^(improves|improved|improving)\b",
    r"^(changes|changed|changing)\b"
]


def validate_commit_message(
    message: str,
    strict: bool = False,
    max_header_length: int = 72,
    max_line_length: int = 100
) -> ValidationResult:
    """
    Validate a commit message against conventional commits spec and best practices.

    Args:
        message: Full commit message to validate
        strict: If True, warnings are treated as errors
        max_header_length: Maximum length for header (default: 72)
        max_line_length: Maximum length for body lines (default: 100)

    Returns:
        ValidationResult with issues and validity status
    """
    issues: List[ValidationIssue] = []

    if not message or not message.strip():
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Commit message is empty"
        ))
        return ValidationResult(valid=False, issues=issues)

    # Split into lines
    lines = message.split("\n")
    header = lines[0].strip()

    # 1. Parse conventional commit format
    parsed = parse_conventional_commit(message)

    if not parsed:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Not a valid conventional commit format",
            line=1,
            suggestion="Use format: type(scope): description"
        ))
        return ValidationResult(valid=False, issues=issues, parsed_commit=None)

    # 2. Validate header length
    if len(header) > max_header_length:
        severity = ValidationSeverity.ERROR if strict else ValidationSeverity.WARNING
        issues.append(ValidationIssue(
            severity=severity,
            message=f"Header too long ({len(header)} chars, recommended max: {max_header_length})",
            line=1,
            suggestion="Keep header concise and move details to body"
        ))

    # 3. Validate description
    description = parsed.description

    # Check if description starts with lowercase
    if description and description[0].isupper():
        severity = ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR
        issues.append(ValidationIssue(
            severity=severity,
            message="Description should start with lowercase",
            line=1,
            suggestion=f"Use: {description[0].lower()}{description[1:]}"
        ))

    # Check for trailing period
    if description.endswith("."):
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Description should not end with period",
            line=1,
            suggestion=description.rstrip(".")
        ))

    # Check imperative mood
    first_word = description.split()[0].lower() if description else ""

    # Check for non-imperative patterns
    is_non_imperative = any(
        re.match(pattern, description, re.IGNORECASE)
        for pattern in NON_IMPERATIVE_PATTERNS
    )

    if is_non_imperative:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Use imperative mood (e.g., 'add' not 'added' or 'adds')",
            line=1,
            suggestion="Rewrite description in imperative mood"
        ))
    elif first_word not in IMPERATIVE_VERBS:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            message=f"Consider starting with imperative verb (add, fix, update, etc.)",
            line=1
        ))

    # 4. Validate body (if present)
    if len(lines) > 1:
        # Check for blank line after header
        if lines[1].strip() != "":
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Missing blank line between header and body",
                line=2,
                suggestion="Add empty line after header"
            ))

        # Check body line lengths
        for i, line in enumerate(lines[2:], start=3):
            if len(line) > max_line_length:
                severity = ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR
                issues.append(ValidationIssue(
                    severity=severity,
                    message=f"Line too long ({len(line)} chars, max: {max_line_length})",
                    line=i,
                    suggestion="Wrap long lines"
                ))

    # 5. Check scope format
    if parsed.scope:
        # Scope should be lowercase, alphanumeric with hyphens
        if not re.match(r"^[a-z0-9-]+$", parsed.scope):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Scope should be lowercase alphanumeric with hyphens",
                line=1,
                suggestion=re.sub(r"[^a-z0-9-]", "-", parsed.scope.lower())
            ))

    # 6. Breaking change validation
    if parsed.breaking:
        # If marked as breaking, should have BREAKING CHANGE: in footer
        if not parsed.footer or "BREAKING CHANGE:" not in parsed.footer:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Breaking change marker (!) used but no BREAKING CHANGE: footer",
                suggestion="Add 'BREAKING CHANGE: description' in footer"
            ))

    # Determine overall validity
    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
    valid = not has_errors

    return ValidationResult(
        valid=valid,
        issues=issues,
        parsed_commit=parsed
    )


def suggest_improvements(message: str) -> List[str]:
    """
    Suggest improvements for a commit message.

    Args:
        message: Commit message to analyze

    Returns:
        List of improvement suggestions
    """
    suggestions = []

    # Validate first
    result = validate_commit_message(message, strict=False)

    if not result.valid:
        suggestions.append("Fix validation errors first")
        return suggestions

    # Check for common improvements
    parsed = result.parsed_commit
    if not parsed:
        return suggestions

    # Suggest adding scope if missing
    if not parsed.scope and len(message.split("\n")[0]) < 50:
        suggestions.append("Consider adding a scope to provide context: type(scope): description")

    # Suggest adding body for complex changes
    if not parsed.body and not parsed.footer:
        suggestions.append("Consider adding a body to explain why the change was made")

    # Check for issue references
    if not parsed.footer or not re.search(r"#\d+", parsed.footer):
        suggestions.append("Consider referencing related issues (e.g., 'Closes #123' in footer)")

    return suggestions


def is_conventional_commit(message: str) -> bool:
    """
    Quick check if message follows conventional commit format.

    Args:
        message: Commit message to check

    Returns:
        True if valid conventional commit
    """
    return parse_conventional_commit(message) is not None


def get_commit_type(message: str) -> Optional[CommitType]:
    """
    Extract commit type from message.

    Args:
        message: Commit message

    Returns:
        CommitType or None if not valid
    """
    parsed = parse_conventional_commit(message)
    return parsed.type if parsed else None
