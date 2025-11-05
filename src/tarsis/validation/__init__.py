"""
Validation module for testing and code quality checks.

Provides multi-tier validation:
1. Automated tests (preferred)
2. Static analysis (fallback)
3. Linting (second fallback)
4. Syntax checking (always available)
5. Dependency validation (supplementary)

Fully implemented! ✅
- Test framework detection ✅
- Validation tier detection ✅
- Test execution ✅
- Static analysis ✅
- Linting ✅
- Syntax checking ✅
- User interaction for repos without tests ✅
- Validation orchestration ✅
- Result reporting ✅
- Dependency/import validation ✅
"""

# Detection
from .detector import (
    TestFrameworkDetector,
    ValidationTierDetector,
    TestDetectionResult,
    ValidationTier
)

# Result types
from .result_types import (
    ValidationResult,
    ValidationStatus,
    TestResult,
    AnalysisResult,
    LintResult,
    SyntaxResult,
    TestFailure,
    AnalysisIssue,
    LintIssue,
    SyntaxError
)

# User interaction
from .no_tests_handler import (
    NoTestsHandler,
    NoTestsDecision,
    NoTestsConfig
)

# Orchestration
from .orchestrator import ValidationOrchestrator

# Reporting
from .reporter import ValidationReporter

# Execution
from .runner import TestRunner
from .static_analyzer import StaticAnalyzer
from .linter import Linter
from .syntax_checker import SyntaxChecker

# Dependency validation
from .dependency_validator import (
    DependencyValidator,
    DependencyResult,
    DependencyIssue
)

__all__ = [
    # Detection
    "TestFrameworkDetector",
    "ValidationTierDetector",
    "TestDetectionResult",
    "ValidationTier",
    # Result types
    "ValidationResult",
    "ValidationStatus",
    "TestResult",
    "AnalysisResult",
    "LintResult",
    "SyntaxResult",
    "TestFailure",
    "AnalysisIssue",
    "LintIssue",
    "SyntaxError",
    # User interaction
    "NoTestsHandler",
    "NoTestsDecision",
    "NoTestsConfig",
    # Orchestration
    "ValidationOrchestrator",
    # Reporting
    "ValidationReporter",
    # Execution
    "TestRunner",
    "StaticAnalyzer",
    "Linter",
    "SyntaxChecker",
    # Dependency validation
    "DependencyValidator",
    "DependencyResult",
    "DependencyIssue",
]
