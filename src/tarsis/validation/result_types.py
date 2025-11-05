"""
Result types for validation operations.

Defines data structures for test results, static analysis, linting, and unified validation results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from .detector import ValidationTier


class ValidationStatus(Enum):
    """Status of validation"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestFailure:
    """Information about a test failure"""
    test_name: str
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    traceback: Optional[str] = None


@dataclass
class TestResult:
    """Result of test execution"""
    status: ValidationStatus
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    duration: float = 0.0
    failures: List[TestFailure] = field(default_factory=list)
    output: str = ""
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Check if all tests passed"""
        return self.status == ValidationStatus.PASSED and self.failed_tests == 0


@dataclass
class AnalysisIssue:
    """Issue found during static analysis"""
    severity: str  # error, warning, info
    message: str
    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None  # Error code (e.g., "E501" for flake8)


@dataclass
class AnalysisResult:
    """Result of static analysis (type checking, etc.)"""
    status: ValidationStatus
    tool: str  # mypy, tsc, pyright, flow, etc.
    issues: List[AnalysisIssue] = field(default_factory=list)
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    output: str = ""
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Check if analysis passed (no errors)"""
        return self.status == ValidationStatus.PASSED and self.errors == 0


@dataclass
class LintIssue:
    """Issue found during linting"""
    severity: str  # error, warning, convention, refactor
    message: str
    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    rule: Optional[str] = None  # Rule name (e.g., "no-unused-vars")


@dataclass
class LintResult:
    """Result of linting"""
    status: ValidationStatus
    tool: str  # pylint, flake8, eslint, rubocop, etc.
    issues: List[LintIssue] = field(default_factory=list)
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    output: str = ""
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Check if linting passed (configurable threshold)"""
        # For now, consider it passed if no errors (warnings are OK)
        return self.status == ValidationStatus.PASSED and self.errors == 0


@dataclass
class SyntaxError:
    """Syntax error in a file"""
    file_path: str
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None


@dataclass
class SyntaxResult:
    """Result of syntax checking"""
    status: ValidationStatus
    errors: List[SyntaxError] = field(default_factory=list)
    total_errors: int = 0
    files_checked: int = 0
    output: str = ""
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Check if syntax is valid"""
        return self.status == ValidationStatus.PASSED and self.total_errors == 0


@dataclass
class ValidationResult:
    """Unified validation result combining all validation tiers"""

    # Overall status
    status: ValidationStatus
    tier_used: ValidationTier  # Which validation tier was actually used

    # Individual results (optional, depending on what was run)
    test_result: Optional[TestResult] = None
    analysis_result: Optional[AnalysisResult] = None
    lint_result: Optional[LintResult] = None
    syntax_result: Optional[SyntaxResult] = None

    # Summary
    summary: str = ""
    details: str = ""

    # Metadata
    duration: float = 0.0
    timestamp: Optional[str] = None

    # User interaction
    user_decision: Optional[str] = None  # If user was asked for decision

    @property
    def passed(self) -> bool:
        """Check if validation passed overall"""
        if self.status == ValidationStatus.PASSED:
            # Check the result that was actually used
            if self.test_result:
                return self.test_result.passed
            elif self.analysis_result:
                return self.analysis_result.passed
            elif self.lint_result:
                return self.lint_result.passed
            elif self.syntax_result:
                return self.syntax_result.passed
            return True
        return False

    @property
    def has_failures(self) -> bool:
        """Check if there are any failures"""
        return not self.passed

    def get_failure_summary(self) -> str:
        """Get a summary of failures"""
        failures = []

        if self.test_result and not self.test_result.passed:
            failures.append(f"Tests: {self.test_result.failed_tests} failed")

        if self.analysis_result and not self.analysis_result.passed:
            failures.append(f"Static Analysis: {self.analysis_result.errors} errors")

        if self.lint_result and not self.lint_result.passed:
            failures.append(f"Linting: {self.lint_result.errors} errors")

        if self.syntax_result and not self.syntax_result.passed:
            failures.append(f"Syntax: {self.syntax_result.total_errors} errors")

        return ", ".join(failures) if failures else "No failures"
