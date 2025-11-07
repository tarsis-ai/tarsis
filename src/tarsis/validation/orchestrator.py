"""
Validation orchestrator.

Coordinates the entire validation workflow including test detection, user interaction,
execution, and fallback validation tiers.
"""

import time
from typing import Optional, List, Callable, Awaitable, Any
from datetime import datetime
from pathlib import Path

from .detector import (
    TestFrameworkDetector,
    ValidationTierDetector,
    TestDetectionResult,
    ValidationTier
)
from .no_tests_handler import NoTestsHandler, NoTestsDecision, NoTestsConfig
from .result_types import (
    ValidationResult,
    ValidationStatus,
    TestResult,
    AnalysisResult,
    LintResult,
    SyntaxResult
)
from .runner import TestRunner
from .static_analyzer import StaticAnalyzer
from .linter import Linter
from .syntax_checker import SyntaxChecker
from .dependency_validator import DependencyValidator, DependencyResult


class ValidationOrchestrator:
    """
    Orchestrates the validation workflow.

    Handles:
    1. Test framework detection
    2. User interaction when no tests found
    3. Execution of validation (tests, static analysis, linting, syntax)
    4. Fallback chain management
    5. Result aggregation
    """

    def __init__(
        self,
        repo_path: str,
        no_tests_config: Optional[NoTestsConfig] = None,
        ask_followup_callback: Optional[Callable[[dict], Awaitable[str]]] = None
    ):
        """
        Initialize orchestrator.

        Args:
            repo_path: Path to repository
            no_tests_config: Configuration for handling repos without tests
            ask_followup_callback: Async callback to ask user questions
                                  (takes question dict, returns user response)
        """
        self.repo_path = Path(repo_path)
        self.no_tests_handler = NoTestsHandler(no_tests_config)
        self.ask_followup_callback = ask_followup_callback

    async def validate(
        self,
        modified_files: Optional[List[str]] = None,
        check_dependencies: bool = False
    ) -> ValidationResult:
        """
        Run validation on the repository or specific files.

        Args:
            modified_files: Optional list of files that were modified (for targeted validation)
            check_dependencies: Whether to run dependency/import validation (default: False)

        Returns:
            ValidationResult with outcomes
        """
        start_time = time.time()

        # Step 1: Detect test framework and available tiers
        detection_result = self._detect_tests_and_tiers()

        # Step 2: Determine validation strategy
        if detection_result.has_tests:
            # Has tests - run them
            result = await self._run_tests(detection_result, modified_files)
        else:
            # No tests - handle appropriately
            result = await self._handle_no_tests(detection_result, modified_files)

        # Step 3: Optionally check dependencies (supplementary check)
        if check_dependencies and detection_result.language:
            dep_result = await self._check_dependencies(detection_result, modified_files)
            # Add dependency info to details
            if dep_result and not dep_result.passed:
                result.details += f"\n\n**Dependency Check:**\n"
                result.details += f"- {dep_result.errors} error(s), {dep_result.warnings} warning(s)\n"
                if dep_result.issues:
                    result.details += "Issues:\n"
                    for issue in dep_result.issues[:5]:
                        result.details += f"  - [{issue.severity}] {issue.message}\n"

        # Set metadata
        result.duration = time.time() - start_time
        result.timestamp = datetime.utcnow().isoformat()

        return result

    def _detect_tests_and_tiers(self) -> TestDetectionResult:
        """
        Detect test framework and available validation tiers.

        Returns:
            TestDetectionResult with detection info
        """
        # Detect test framework
        detector = TestFrameworkDetector(str(self.repo_path))
        detection_result = detector.detect()

        # Detect available fallback tiers
        tier_detector = ValidationTierDetector(
            str(self.repo_path),
            language=detection_result.language
        )
        detection_result.available_tiers = tier_detector.detect_available_tiers()

        return detection_result

    async def _run_tests(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Run tests using detected framework.

        Args:
            detection_result: Test detection results
            modified_files: Optional list of modified files

        Returns:
            ValidationResult with test outcomes
        """
        # Create test runner and execute tests
        runner = TestRunner(str(self.repo_path))
        test_result = await runner.run_tests(detection_result, modified_files)

        # Check if no tests were found (0 total tests)
        if test_result.total_tests == 0:
            # No tests found - fall back to other validation tiers
            return await self._run_fallback_validation(
                detection_result,
                modified_files,
                user_decision=f"No tests found - using fallback validation (syntax checking)"
            )

        # If test execution failed (e.g., pytest not installed), fall back to other validation tiers
        if test_result.status == ValidationStatus.ERROR:
            # Check if this is a "tool not found" error (pytest, jest, etc. not installed)
            error_msg = test_result.error_message or ""
            is_tool_missing = any(phrase in error_msg.lower() for phrase in [
                "no such file or directory",
                "command not found",
                "not found",
                "cannot find"
            ])

            if is_tool_missing:
                # Test framework detected but tool not installed - fall back to other tiers
                return await self._run_fallback_validation(
                    detection_result,
                    modified_files,
                    user_decision=f"Tests detected but {detection_result.framework} not installed - using fallback validation"
                )

        # Determine overall status
        if test_result.status == ValidationStatus.ERROR:
            status = ValidationStatus.ERROR
            summary = f"Error running tests: {test_result.error_message}"
        elif test_result.passed:
            status = ValidationStatus.PASSED
            summary = f"All tests passed! {test_result.passed_tests}/{test_result.total_tests} tests passed"
        else:
            status = ValidationStatus.FAILED
            summary = f"Tests failed: {test_result.failed_tests}/{test_result.total_tests} tests failed"

        details = f"Framework: {detection_result.framework}\n"
        details += f"Command: {detection_result.test_command}\n"
        if test_result.failures:
            details += f"\nFailures:\n"
            for failure in test_result.failures[:5]:  # Show first 5
                details += f"- {failure.test_name}: {failure.error_message}\n"

        return ValidationResult(
            status=status,
            tier_used=ValidationTier.TESTS,
            test_result=test_result,
            summary=summary,
            details=details
        )

    async def _handle_no_tests(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Handle the scenario when no tests are found.

        Args:
            detection_result: Test detection results
            modified_files: Optional list of modified files

        Returns:
            ValidationResult based on user decision and fallback validation
        """
        # Check if we should ask the user
        should_ask = self.no_tests_handler.should_ask_user(detection_result)

        user_decision = None
        decision = None

        if should_ask and self.ask_followup_callback:
            # Ask the user what to do
            question_data = self.no_tests_handler.create_question_for_tool(detection_result)
            user_response = await self.ask_followup_callback(question_data)

            # Parse response
            decision = self.no_tests_handler.parse_user_response(
                user_response,
                question_data["options"]
            )
            user_decision = f"{decision.value} (user chose: {user_response})"
        else:
            # Use default behavior
            decision = self.no_tests_handler.get_default_decision(detection_result)
            user_decision = f"{decision.value} (default behavior)"

        # Execute based on decision
        if decision == NoTestsDecision.ABORT:
            return ValidationResult(
                status=ValidationStatus.ERROR,
                tier_used=ValidationTier.SYNTAX,
                summary="Task aborted: No tests found and user chose to abort",
                user_decision=user_decision
            )

        elif decision == NoTestsDecision.SKIP_VALIDATION:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                tier_used=ValidationTier.SYNTAX,
                summary="Validation skipped by user decision",
                details="User chose to skip validation and proceed directly to PR creation.",
                user_decision=user_decision
            )

        elif decision == NoTestsDecision.CREATE_TESTS:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                tier_used=ValidationTier.SYNTAX,
                summary="Test creation requested",
                details="User requested that tests be created before proceeding. This requires manual implementation.",
                user_decision=user_decision
            )

        elif decision == NoTestsDecision.PROCEED_WITH_VALIDATION:
            # Proceed with fallback validation
            return await self._run_fallback_validation(
                detection_result,
                modified_files,
                user_decision
            )

        # Should never reach here
        return ValidationResult(
            status=ValidationStatus.ERROR,
            tier_used=ValidationTier.SYNTAX,
            summary="Unknown decision state",
            user_decision=user_decision
        )

    async def _run_fallback_validation(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]],
        user_decision: Optional[str]
    ) -> ValidationResult:
        """
        Run fallback validation tiers.

        Args:
            detection_result: Test detection results
            modified_files: Optional list of modified files
            user_decision: User's decision (for tracking)

        Returns:
            ValidationResult from the highest available tier
        """
        available_tiers = detection_result.available_tiers

        if not available_tiers:
            # Only syntax checking available
            return await self._run_syntax_check(modified_files, user_decision)

        # Try tiers in order of preference
        for tier in available_tiers:
            if tier == ValidationTier.STATIC_ANALYSIS:
                result = await self._run_static_analysis(detection_result, modified_files)
                if result.status != ValidationStatus.ERROR:
                    result.user_decision = user_decision
                    return result

            elif tier == ValidationTier.LINTING:
                result = await self._run_linting(detection_result, modified_files)
                if result.status != ValidationStatus.ERROR:
                    result.user_decision = user_decision
                    return result

        # Fall back to syntax checking
        result = await self._run_syntax_check(modified_files, user_decision)
        result.user_decision = user_decision
        return result

    async def _run_static_analysis(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Run static analysis.

        Args:
            detection_result: Test detection results
            modified_files: Optional list of modified files

        Returns:
            ValidationResult with static analysis outcomes
        """
        # Create static analyzer and run
        analyzer = StaticAnalyzer(str(self.repo_path))
        analysis_result = await analyzer.run_analysis(
            detection_result.language,
            modified_files
        )

        # Determine overall status
        if analysis_result.status == ValidationStatus.ERROR:
            status = ValidationStatus.ERROR
            summary = f"Error running static analysis: {analysis_result.error_message}"
        elif analysis_result.status == ValidationStatus.SKIPPED:
            status = ValidationStatus.SKIPPED
            summary = analysis_result.output
        elif analysis_result.passed:
            status = ValidationStatus.PASSED
            summary = f"Static analysis passed! {analysis_result.tool} found no errors"
        else:
            status = ValidationStatus.FAILED
            summary = f"Static analysis failed: {analysis_result.errors} error(s), {analysis_result.warnings} warning(s) found by {analysis_result.tool}"

        details = f"Tool: {analysis_result.tool}\n"
        details += f"Language: {detection_result.language}\n"
        if analysis_result.issues:
            details += f"\nTop issues:\n"
            for issue in analysis_result.issues[:5]:
                details += f"- {issue.file_path}:{issue.line_number} [{issue.severity}] {issue.message}\n"

        return ValidationResult(
            status=status,
            tier_used=ValidationTier.STATIC_ANALYSIS,
            analysis_result=analysis_result,
            summary=summary,
            details=details
        )

    async def _run_linting(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Run linting.

        Args:
            detection_result: Test detection results
            modified_files: Optional list of modified files

        Returns:
            ValidationResult with linting outcomes
        """
        # Create linter and run
        linter = Linter(str(self.repo_path))
        lint_result = await linter.run_linting(
            detection_result.language,
            modified_files
        )

        # Determine overall status
        if lint_result.status == ValidationStatus.ERROR:
            status = ValidationStatus.ERROR
            summary = f"Error running linter: {lint_result.error_message}"
        elif lint_result.status == ValidationStatus.SKIPPED:
            status = ValidationStatus.SKIPPED
            summary = lint_result.output
        elif lint_result.passed:
            status = ValidationStatus.PASSED
            summary = f"Linting passed! {lint_result.tool} found no errors"
        else:
            status = ValidationStatus.FAILED
            summary = f"Linting failed: {lint_result.errors} error(s), {lint_result.warnings} warning(s) found by {lint_result.tool}"

        details = f"Tool: {lint_result.tool}\n"
        details += f"Language: {detection_result.language}\n"
        if lint_result.issues:
            details += f"\nTop issues:\n"
            for issue in lint_result.issues[:5]:
                details += f"- {issue.file_path}:{issue.line_number} [{issue.severity}] {issue.message}\n"

        return ValidationResult(
            status=status,
            tier_used=ValidationTier.LINTING,
            lint_result=lint_result,
            summary=summary,
            details=details
        )

    async def _run_syntax_check(
        self,
        modified_files: Optional[List[str]] = None,
        user_decision: Optional[str] = None
    ) -> ValidationResult:
        """
        Run syntax checking.

        Args:
            modified_files: Optional list of modified files
            user_decision: User's decision (for tracking)

        Returns:
            ValidationResult with syntax check outcomes
        """
        # Create syntax checker and run
        checker = SyntaxChecker(str(self.repo_path))
        syntax_result = await checker.check_syntax(modified_files)

        # Determine overall status
        if syntax_result.status == ValidationStatus.ERROR:
            status = ValidationStatus.ERROR
            summary = f"Error checking syntax: {syntax_result.error_message}"
        elif syntax_result.status == ValidationStatus.SKIPPED:
            status = ValidationStatus.SKIPPED
            summary = syntax_result.output
        elif syntax_result.passed:
            status = ValidationStatus.PASSED
            summary = f"Syntax check passed! No syntax errors in {syntax_result.files_checked} file(s)"
        else:
            status = ValidationStatus.FAILED
            summary = f"Syntax check failed: {syntax_result.total_errors} error(s) found in {syntax_result.files_checked} file(s)"

        details = f"Files checked: {syntax_result.files_checked}\n"
        if syntax_result.errors:
            details += f"\nSyntax errors:\n"
            for error in syntax_result.errors[:5]:
                loc = f"{error.file_path}"
                if error.line_number:
                    loc += f":{error.line_number}"
                details += f"- {loc}: {error.message}\n"

        return ValidationResult(
            status=status,
            tier_used=ValidationTier.SYNTAX,
            syntax_result=syntax_result,
            summary=summary,
            details=details,
            user_decision=user_decision
        )

    async def _check_dependencies(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]] = None
    ) -> Optional[DependencyResult]:
        """
        Check dependencies and imports (supplementary validation).

        Args:
            detection_result: Test detection results
            modified_files: Optional list of modified files

        Returns:
            DependencyResult or None if skipped
        """
        try:
            validator = DependencyValidator(str(self.repo_path))
            result = await validator.validate_dependencies(
                detection_result.language,
                modified_files
            )
            return result
        except Exception as e:
            # Don't fail the whole validation if dependency check fails
            return None
