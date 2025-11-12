"""
Test runner for executing tests across multiple test frameworks.

Supports pytest, unittest, jest, mocha, vitest, go test, cargo test, rspec, and junit.
"""

import subprocess
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .result_types import TestResult, TestFailure, ValidationStatus
from .detector import TestDetectionResult


class TestRunner:
    """
    Executes tests using detected test framework and parses results.

    Supports multiple test frameworks across different languages.
    """

    def __init__(self, repo_path: str):
        """
        Initialize test runner.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)

    async def run_tests(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]] = None
    ) -> TestResult:
        """
        Run tests using detected framework.

        Args:
            detection_result: Test framework detection results
            modified_files: Optional list of modified files for targeted testing

        Returns:
            TestResult with test execution outcomes
        """
        if not detection_result.framework or not detection_result.test_command:
            return TestResult(
                status=ValidationStatus.SKIPPED,
                output="No test framework detected"
            )

        start_time = time.time()

        try:
            # Build test command
            command = self._build_test_command(
                detection_result,
                modified_files
            )

            # Execute tests
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.repo_path),
                shell=False
            )

            duration = time.time() - start_time

            # Parse output based on framework
            test_result = self._parse_test_output(
                detection_result.framework,
                result.stdout,
                result.stderr,
                result.returncode,
                duration
            )

            return test_result

        except subprocess.TimeoutExpired:
            return TestResult(
                status=ValidationStatus.ERROR,
                error_message="Test execution timed out (5 minutes)",
                duration=time.time() - start_time
            )
        except Exception as e:
            return TestResult(
                status=ValidationStatus.ERROR,
                error_message=f"Failed to run tests: {str(e)}",
                duration=time.time() - start_time,
                output=str(e)
            )

    def _build_test_command(
        self,
        detection_result: TestDetectionResult,
        modified_files: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build test command based on framework.

        Args:
            detection_result: Detection results
            modified_files: Optional modified files for targeted testing

        Returns:
            Command as list of strings
        """
        framework = detection_result.framework
        base_command = detection_result.test_command

        # Parse base command
        if isinstance(base_command, str):
            command = base_command.split()
        else:
            command = list(base_command)

        # Add framework-specific flags for better output parsing
        if framework == "pytest":
            # Add flags for better output
            if "--tb=short" not in command:
                command.append("--tb=short")
            if "-v" not in command:
                command.append("-v")
            # For targeted testing
            if modified_files:
                test_files = self._find_related_test_files(modified_files, detection_result)
                if test_files:
                    command.extend(test_files)

        elif framework == "jest":
            # Use JSON output if possible
            if "--json" not in command and "--verbose" not in command:
                command.append("--verbose")
            # Targeted testing
            if modified_files:
                # Jest can find related tests automatically
                command.append("--findRelatedTests")
                command.extend(modified_files)

        elif framework == "go_test":
            # Add verbose flag
            if "-v" not in command:
                command.append("-v")

        elif framework == "cargo_test":
            # Cargo test is already verbose by default
            pass

        elif framework == "rspec":
            # Add format flag
            if "--format" not in " ".join(command):
                command.extend(["--format", "documentation"])

        return command

    def _parse_test_output(
        self,
        framework: str,
        stdout: str,
        stderr: str,
        return_code: int,
        duration: float
    ) -> TestResult:
        """
        Parse test output based on framework.

        Args:
            framework: Test framework name
            stdout: Standard output
            stderr: Standard error
            return_code: Process return code
            duration: Test duration

        Returns:
            Parsed TestResult
        """
        output = stdout + "\n" + stderr

        # Try JSON parsing first (for supported frameworks)
        if framework in ("jest", "go_test") and "{" in output:
            json_result = self._try_parse_json_output(framework, output)
            if json_result:
                json_result.duration = duration
                json_result.output = output
                return json_result

        # Framework-specific parsers
        if framework in ("pytest", "unittest"):
            return self._parse_pytest_output(stdout, stderr, return_code, duration)
        elif framework in ("jest", "mocha", "vitest"):
            return self._parse_jest_output(stdout, stderr, return_code, duration)
        elif framework == "go_test":
            return self._parse_go_test_output(stdout, stderr, return_code, duration)
        elif framework == "cargo_test":
            return self._parse_cargo_output(stdout, stderr, return_code, duration)
        elif framework == "rspec":
            return self._parse_rspec_output(stdout, stderr, return_code, duration)
        else:
            # Generic parser
            return self._parse_generic_output(stdout, stderr, return_code, duration)

    def _try_parse_json_output(self, framework: str, output: str) -> Optional[TestResult]:
        """Try to parse JSON output from test frameworks that support it"""
        try:
            # Find JSON in output (it might be mixed with other text)
            json_start = output.find('{')
            if json_start == -1:
                return None

            json_data = json.loads(output[json_start:])

            if framework == "go_test" and "Action" in json_data:
                # Go test JSON format
                # This is line-delimited JSON, need different parsing
                return None  # Fall back to text parsing

            return None  # Not a complete JSON format we recognize

        except (json.JSONDecodeError, KeyError):
            return None

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        duration: float
    ) -> TestResult:
        """Parse pytest output"""
        # pytest summary line can be in different orders:
        # "====== X passed in Z.ZZs ======" (all passed)
        # "====== X failed, Y passed in Z.ZZs ======" (some failed)
        # "====== X passed, Y failed in Z.ZZs ======" (some failed, alternate order)
        # "====== X passed, Y failed, Z skipped in Z.ZZs ======" (with skipped)

        # Extract all components separately
        passed_match = re.search(r'(\d+)\s+passed', stdout)
        failed_match = re.search(r'(\d+)\s+failed', stdout)
        skipped_match = re.search(r'(\d+)\s+skipped', stdout)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        skipped = int(skipped_match.group(1)) if skipped_match else 0

        total = passed + failed + skipped

        # Parse failures
        failures = self._parse_pytest_failures(stdout)

        # Determine status
        if return_code == 0 and failed == 0:
            status = ValidationStatus.PASSED
        elif return_code != 0 or failed > 0:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return TestResult(
            status=status,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            duration=duration,
            failures=failures,
            output=stdout + "\n" + stderr
        )

    def _parse_pytest_failures(self, output: str) -> List[TestFailure]:
        """Parse pytest failure details"""
        failures = []

        # Pattern: FAILED test_file.py::test_name - Error message
        failure_pattern = r'FAILED\s+([^:]+)::([^\s]+)\s*-?\s*(.+)?'

        for line in output.split('\n'):
            match = re.search(failure_pattern, line)
            if match:
                file_path = match.group(1)
                test_name = match.group(2)
                error_msg = match.group(3) if match.group(3) else "Test failed"

                failures.append(TestFailure(
                    test_name=f"{file_path}::{test_name}",
                    error_message=error_msg.strip(),
                    file_path=file_path
                ))

        return failures

    def _parse_jest_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        duration: float
    ) -> TestResult:
        """Parse jest/mocha/vitest output"""
        # Jest summary: "Tests:       X failed, Y passed, Z total"
        passed_match = re.search(r'(\d+)\s+passed', stdout)
        failed_match = re.search(r'(\d+)\s+failed', stdout)
        skipped_match = re.search(r'(\d+)\s+skipped', stdout)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        skipped = int(skipped_match.group(1)) if skipped_match else 0
        total = passed + failed + skipped

        # Parse failures
        failures = self._parse_jest_failures(stdout)

        status = ValidationStatus.PASSED if return_code == 0 and failed == 0 else ValidationStatus.FAILED

        return TestResult(
            status=status,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            duration=duration,
            failures=failures,
            output=stdout + "\n" + stderr
        )

    def _parse_jest_failures(self, output: str) -> List[TestFailure]:
        """Parse jest failure details"""
        failures = []

        # Pattern: ● test suite › test name
        current_test = None
        current_error = []

        for line in output.split('\n'):
            if line.strip().startswith('●'):
                # Save previous test if any
                if current_test and current_error:
                    failures.append(TestFailure(
                        test_name=current_test,
                        error_message='\n'.join(current_error),
                    ))

                # Start new test
                current_test = line.strip('● ').strip()
                current_error = []
            elif current_test and line.strip() and not line.startswith('  '):
                # Error message line
                current_error.append(line.strip())

        # Don't forget last test
        if current_test and current_error:
            failures.append(TestFailure(
                test_name=current_test,
                error_message='\n'.join(current_error)
            ))

        return failures

    def _parse_go_test_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        duration: float
    ) -> TestResult:
        """Parse go test output"""
        # Go test output: --- PASS/FAIL: TestName
        passed = len(re.findall(r'---\s*PASS:', stdout))
        failed = len(re.findall(r'---\s*FAIL:', stdout))
        total = passed + failed

        # Parse failures
        failures = self._parse_go_failures(stdout)

        status = ValidationStatus.PASSED if return_code == 0 and failed == 0 else ValidationStatus.FAILED

        return TestResult(
            status=status,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            duration=duration,
            failures=failures,
            output=stdout + "\n" + stderr
        )

    def _parse_go_failures(self, output: str) -> List[TestFailure]:
        """Parse go test failures"""
        failures = []
        current_test = None
        error_lines = []

        for line in output.split('\n'):
            if '--- FAIL:' in line:
                # Save previous
                if current_test:
                    failures.append(TestFailure(
                        test_name=current_test,
                        error_message='\n'.join(error_lines)
                    ))

                # New test
                match = re.search(r'FAIL:\s+(\S+)', line)
                current_test = match.group(1) if match else "Unknown test"
                error_lines = []
            elif current_test and line.strip():
                error_lines.append(line.strip())

        if current_test:
            failures.append(TestFailure(
                test_name=current_test,
                error_message='\n'.join(error_lines)
            ))

        return failures

    def _parse_cargo_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        duration: float
    ) -> TestResult:
        """Parse cargo test output"""
        # Cargo: "test result: ok. X passed; Y failed; Z ignored"
        result_pattern = r'test result:.+?(\d+)\s+passed;\s+(\d+)\s+failed'
        match = re.search(result_pattern, stdout)

        passed = int(match.group(1)) if match else 0
        failed = int(match.group(2)) if match else 0
        total = passed + failed

        # Parse failures
        failures = []
        for line in stdout.split('\n'):
            if line.startswith('test ') and 'FAILED' in line:
                test_name = line.split()[1]
                failures.append(TestFailure(
                    test_name=test_name,
                    error_message="Test failed"
                ))

        status = ValidationStatus.PASSED if return_code == 0 and failed == 0 else ValidationStatus.FAILED

        return TestResult(
            status=status,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            duration=duration,
            failures=failures,
            output=stdout + "\n" + stderr
        )

    def _parse_rspec_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        duration: float
    ) -> TestResult:
        """Parse rspec output"""
        # RSpec: "X examples, Y failures"
        examples_match = re.search(r'(\d+)\s+examples?', stdout)
        failures_match = re.search(r'(\d+)\s+failures?', stdout)

        total = int(examples_match.group(1)) if examples_match else 0
        failed = int(failures_match.group(1)) if failures_match else 0
        passed = total - failed

        # Parse failures (basic)
        failures = []
        if failed > 0:
            failures.append(TestFailure(
                test_name="RSpec tests",
                error_message=f"{failed} example(s) failed"
            ))

        status = ValidationStatus.PASSED if return_code == 0 and failed == 0 else ValidationStatus.FAILED

        return TestResult(
            status=status,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            duration=duration,
            failures=failures,
            output=stdout + "\n" + stderr
        )

    def _parse_generic_output(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        duration: float
    ) -> TestResult:
        """Generic parser for unknown frameworks"""
        # Try to extract any numbers that look like test counts
        passed_match = re.search(r'(\d+)\s+(?:passed|ok|success)', stdout, re.IGNORECASE)
        failed_match = re.search(r'(\d+)\s+(?:failed|error|failure)', stdout, re.IGNORECASE)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        status = ValidationStatus.PASSED if return_code == 0 else ValidationStatus.FAILED

        return TestResult(
            status=status,
            total_tests=passed + failed,
            passed_tests=passed,
            failed_tests=failed,
            duration=duration,
            output=stdout + "\n" + stderr
        )

    def _find_related_test_files(
        self,
        modified_files: List[str],
        detection_result: TestDetectionResult
    ) -> List[str]:
        """Find test files related to modified files"""
        # Simple heuristic: look for test files with similar names
        test_files = []

        for modified_file in modified_files:
            path = Path(modified_file)
            stem = path.stem

            # Look for test files matching this stem
            for test_file in detection_result.test_files:
                if stem in test_file or path.parent.name in test_file:
                    test_files.append(test_file)

        return test_files
