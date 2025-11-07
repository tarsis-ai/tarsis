"""
Conventional Commits implementation for Tarsis.

Provides utilities for detecting, formatting, and validating conventional commit messages
following the Conventional Commits specification (https://www.conventionalcommits.org/).

Standard format: type(scope): description

where:
- type: The kind of change (feat, fix, docs, etc.)
- scope: Optional context (file/module/component affected)
- description: Brief summary in imperative mood
"""

import re
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


class CommitType(Enum):
    """Standard conventional commit types."""

    FEAT = "feat"        # New feature
    FIX = "fix"          # Bug fix
    DOCS = "docs"        # Documentation only changes
    STYLE = "style"      # Code style/formatting (no logic change)
    REFACTOR = "refactor"  # Code restructuring (no behavior change)
    TEST = "test"        # Adding or updating tests
    CHORE = "chore"      # Maintenance tasks, dependencies
    PERF = "perf"        # Performance improvements
    BUILD = "build"      # Build system or external dependencies
    CI = "ci"            # CI/CD configuration changes
    REVERT = "revert"    # Reverting previous commits

    @property
    def description(self) -> str:
        """Get human-readable description of commit type."""
        descriptions = {
            CommitType.FEAT: "A new feature",
            CommitType.FIX: "A bug fix",
            CommitType.DOCS: "Documentation only changes",
            CommitType.STYLE: "Changes that don't affect code meaning (formatting, etc.)",
            CommitType.REFACTOR: "Code change that neither fixes a bug nor adds a feature",
            CommitType.TEST: "Adding missing tests or correcting existing tests",
            CommitType.CHORE: "Changes to build process or auxiliary tools",
            CommitType.PERF: "A code change that improves performance",
            CommitType.BUILD: "Changes that affect the build system or dependencies",
            CommitType.CI: "Changes to CI/CD configuration files and scripts",
            CommitType.REVERT: "Reverts a previous commit"
        }
        return descriptions[self]


@dataclass
class ConventionalCommit:
    """Represents a parsed conventional commit message."""

    type: CommitType
    scope: Optional[str]
    description: str
    body: Optional[str] = None
    footer: Optional[str] = None
    breaking: bool = False

    def format(self) -> str:
        """Format as conventional commit string."""
        # Build header
        header = self.type.value
        if self.scope:
            header += f"({self.scope})"
        if self.breaking:
            header += "!"
        header += f": {self.description}"

        # Build full message
        parts = [header]
        if self.body:
            parts.append("")
            parts.append(self.body)
        if self.footer:
            parts.append("")
            parts.append(self.footer)

        return "\n".join(parts)


# Type detection patterns based on file paths
FILE_TYPE_PATTERNS = {
    CommitType.DOCS: [
        r"\.md$",
        r"^docs/",
        r"^README",
        r"^CHANGELOG",
        r"^LICENSE",
        r"\.rst$",
        r"\.txt$"
    ],
    CommitType.TEST: [
        r"^tests?/",
        r"_test\.py$",
        r"\.test\.",
        r"\.spec\.",
        r"^test_",
        r"/tests?/"
    ],
    CommitType.CI: [
        r"^\.github/workflows/",
        r"^\.gitlab-ci\.yml$",
        r"^\.circleci/",
        r"^\.travis\.yml$",
        r"^azure-pipelines",
        r"^Jenkinsfile$"
    ],
    CommitType.BUILD: [
        r"^setup\.py$",
        r"^pyproject\.toml$",
        r"^requirements.*\.txt$",
        r"^Pipfile",
        r"^package\.json$",
        r"^package-lock\.json$",
        r"^yarn\.lock$",
        r"^pom\.xml$",
        r"^build\.gradle$",
        r"^Makefile$",
        r"^CMakeLists\.txt$",
        r"^Cargo\.toml$"
    ],
    CommitType.CHORE: [
        r"^\.gitignore$",
        r"^\.env",
        r"^\.editorconfig$",
        r"^\.dockerignore$"
    ]
}

# Content-based patterns for commit type detection
CONTENT_PATTERNS = {
    CommitType.FIX: [
        r"\bfix(es|ed|ing)?\b",
        r"\bbug\b",
        r"\berror\b",
        r"\bissue\b",
        r"\bproblem\b",
        r"\bcrash\b"
    ],
    CommitType.FEAT: [
        r"\badd(s|ed|ing)?\b",
        r"\bnew\b",
        r"\bfeature\b",
        r"\bimplement(s|ed|ing)?\b",
        r"\bintroduc(e|es|ed|ing)\b"
    ],
    CommitType.REFACTOR: [
        r"\brefactor(s|ed|ing)?\b",
        r"\brestructur(e|es|ed|ing)\b",
        r"\breorganiz(e|es|ed|ing)\b",
        r"\bsimplif(y|ies|ied|ying)\b",
        r"\bclean(s|ed|ing)?\s+up\b"
    ],
    CommitType.PERF: [
        r"\bperformance\b",
        r"\boptimiz(e|es|ed|ing)\b",
        r"\bfaster\b",
        r"\bspeed\s+up\b",
        r"\bcach(e|es|ed|ing)\b"
    ]
}


def detect_commit_type_from_files(file_paths: List[str]) -> Optional[CommitType]:
    """
    Detect commit type based on modified file paths.

    Args:
        file_paths: List of file paths that were modified

    Returns:
        Detected CommitType or None if no clear type detected
    """
    if not file_paths:
        return None

    # Count matches per type
    type_scores: Dict[CommitType, int] = {t: 0 for t in CommitType}

    for file_path in file_paths:
        for commit_type, patterns in FILE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    type_scores[commit_type] += 1

    # Return type with highest score (if > 0)
    max_score = max(type_scores.values())
    if max_score > 0:
        for commit_type, score in type_scores.items():
            if score == max_score:
                return commit_type

    return None


def detect_commit_type_from_content(description: str) -> Optional[CommitType]:
    """
    Detect commit type based on commit message content.

    Args:
        description: Commit message description

    Returns:
        Detected CommitType or None if no clear type detected
    """
    if not description:
        return None

    description_lower = description.lower()

    # Check content patterns
    for commit_type, patterns in CONTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, description_lower):
                return commit_type

    return None


def detect_scope_from_files(file_paths: List[str]) -> Optional[str]:
    """
    Detect appropriate scope based on modified files.

    Attempts to find common directory or module name.

    Args:
        file_paths: List of file paths that were modified

    Returns:
        Detected scope or None
    """
    if not file_paths:
        return None

    # If single file, use its directory or module name
    if len(file_paths) == 1:
        path = Path(file_paths[0])

        # Skip root-level files
        if len(path.parts) == 1:
            return None

        # Use first directory component
        return path.parts[0]

    # For multiple files, find common prefix
    common_parts = Path(file_paths[0]).parts

    for file_path in file_paths[1:]:
        parts = Path(file_path).parts

        # Find common prefix
        common_parts = tuple(
            p1 for p1, p2 in zip(common_parts, parts)
            if p1 == p2
        )

        if not common_parts:
            break

    # Return first common directory (if exists)
    if common_parts and len(common_parts) > 0:
        return common_parts[0]

    return None


def parse_conventional_commit(message: str) -> Optional[ConventionalCommit]:
    """
    Parse a conventional commit message.

    Args:
        message: Full commit message

    Returns:
        ConventionalCommit object or None if not valid format
    """
    # Split into header and body
    lines = message.split("\n")
    header = lines[0].strip()

    # Parse header: type(scope)!: description
    # Pattern: ^(type)(\(scope\))?(!)?: (.+)$
    pattern = r"^([a-z]+)(?:\(([^)]+)\))?(!)?: (.+)$"
    match = re.match(pattern, header)

    if not match:
        return None

    type_str, scope, breaking_marker, description = match.groups()

    # Validate type
    try:
        commit_type = CommitType(type_str)
    except ValueError:
        return None

    # Extract body and footer
    body = None
    footer = None

    if len(lines) > 2:
        # Find empty line separator
        try:
            first_empty = lines.index("", 1)
            body_lines = lines[1:first_empty]
            remaining = lines[first_empty+1:]

            if body_lines:
                body = "\n".join(body_lines).strip()

            # Check for footer (starts with token: or BREAKING CHANGE:)
            footer_pattern = r"^[A-Z-]+:"
            footer_lines = []
            for line in remaining:
                if re.match(footer_pattern, line) or footer_lines:
                    footer_lines.append(line)

            if footer_lines:
                footer = "\n".join(footer_lines).strip()
        except ValueError:
            # No empty line, treat rest as body
            body = "\n".join(lines[1:]).strip()

    # Check for breaking change
    breaking = bool(breaking_marker) or (footer and "BREAKING CHANGE:" in footer)

    return ConventionalCommit(
        type=commit_type,
        scope=scope,
        description=description,
        body=body,
        footer=footer,
        breaking=breaking
    )


def format_conventional_commit(
    commit_type: CommitType,
    description: str,
    scope: Optional[str] = None,
    body: Optional[str] = None,
    footer: Optional[str] = None,
    breaking: bool = False
) -> str:
    """
    Format a conventional commit message.

    Args:
        commit_type: Type of commit
        description: Brief description in imperative mood
        scope: Optional scope
        body: Optional detailed body
        footer: Optional footer (e.g., BREAKING CHANGE, issue refs)
        breaking: Whether this is a breaking change

    Returns:
        Formatted conventional commit message
    """
    commit = ConventionalCommit(
        type=commit_type,
        scope=scope,
        description=description,
        body=body,
        footer=footer,
        breaking=breaking
    )
    return commit.format()


def is_valid_description(description: str) -> bool:
    """
    Validate commit description follows best practices.

    Checks:
    - Not empty
    - Reasonable length (< 72 chars recommended)
    - Starts with lowercase (conventional)
    - No trailing period

    Args:
        description: Commit description to validate

    Returns:
        True if valid
    """
    if not description or not description.strip():
        return False

    description = description.strip()

    # Check length (warning at 72, error at 100)
    if len(description) > 100:
        return False

    # Should start with lowercase (conventional)
    if description[0].isupper():
        return False

    # Should not end with period
    if description.endswith("."):
        return False

    return True
