"""
Linter for code quality checks.

Supports pylint, flake8, eslint, rubocop, and other linting tools.
"""

import subprocess
import re
import json
from pathlib import Path
from typing import List, Optional, Dict

from .result_types import LintResult, LintIssue, ValidationStatus


class Linter:
    """
    Runs linting tools for code quality checks.

    Fallback Tier 2 when tests and static analysis are not available.
    """

    # Linter configurations
    LINTERS = {
        "python": [
            {
                "name": "pylint",
                "command": ["pylint", "."],
                "config_files": [".pylintrc", "pylintrc", "setup.cfg", "pyproject.toml"],
                "supports_json": True,
            },
            {
                "name": "flake8",
                "command": ["flake8", "."],
                "config_files": [".flake8", "setup.cfg", "tox.ini"],
                "supports_json": False,
            },
        ],
        "javascript": [
            {
                "name": "eslint",
                "command": ["eslint", ".", "--format", "json"],
                "config_files": [".eslintrc", ".eslintrc.json", ".eslintrc.js", ".eslintrc.yml"],
                "supports_json": True,
            },
        ],
        "typescript": [
            {
                "name": "eslint",
                "command": ["eslint", ".", "--format", "json", "--ext", ".ts,.tsx"],
                "config_files": [".eslintrc", ".eslintrc.json", ".eslintrc.js"],
                "supports_json": True,
            },
        ],
        "ruby": [
            {
                "name": "rubocop",
                "command": ["rubocop", "--format", "json"],
                "config_files": [".rubocop.yml"],
                "supports_json": True,
            },
        ],
        "rust": [
            {
                "name": "rustfmt",
                "command": ["cargo", "fmt", "--", "--check"],
                "config_files": ["rustfmt.toml", ".rustfmt.toml"],
                "supports_json": False,
            },
        ],
    }

    def __init__(self, repo_path: str):
        """
        Initialize linter.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)

    async def run_linting(
        self,
        language: str,
        files: Optional[List[str]] = None
    ) -> LintResult:
        """
        Run linting for given language.

        Args:
            language: Programming language
            files: Optional list of files to lint

        Returns:
            LintResult with linting outcomes
        """
        # Get available linters for language
        linters = self.LINTERS.get(language, [])

        if not linters:
            return LintResult(
                status=ValidationStatus.SKIPPED,
                tool="none",
                output=f"No linter available for {language}"
            )

        # Try each linter until one works
        for linter_config in linters:
            tool_name = linter_config["name"]

            # Check if tool is available
            if not self._is_tool_available(tool_name):
                continue

            # Run the linter
            try:
                command = linter_config["command"].copy()

                # Add files if specified
                if files:
                    # Replace "." with specific files for some linters
                    if "." in command and tool_name in ("pylint", "flake8"):
                        command.remove(".")
                        command.extend(files)

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minute timeout
                    cwd=str(self.repo_path)
                )

                # Parse output
                lint_result = self._parse_linter_output(
                    tool_name,
                    result.stdout,
                    result.stderr,
                    result.returncode,
                    linter_config.get("supports_json", False)
                )

                return lint_result

            except subprocess.TimeoutExpired:
                return LintResult(
                    status=ValidationStatus.ERROR,
                    tool=tool_name,
                    error_message="Linting timed out (3 minutes)"
                )
            except FileNotFoundError:
                # Tool not found, try next
                continue
            except Exception as e:
                return LintResult(
                    status=ValidationStatus.ERROR,
                    tool=tool_name,
                    error_message=f"Failed to run {tool_name}: {str(e)}"
                )

        # No linter worked
        return LintResult(
            status=ValidationStatus.SKIPPED,
            tool="none",
            output=f"No working linter found for {language}"
        )

    def _is_tool_available(self, tool_name: str) -> bool:
        """Check if tool is available in PATH"""
        try:
            # Special case for cargo-based tools
            if tool_name == "rustfmt":
                subprocess.run(
                    ["cargo", "fmt", "--version"],
                    capture_output=True,
                    timeout=5
                )
                return True

            subprocess.run(
                [tool_name, "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _parse_linter_output(
        self,
        tool: str,
        stdout: str,
        stderr: str,
        return_code: int,
        supports_json: bool
    ) -> LintResult:
        """
        Parse output from linter.

        Args:
            tool: Tool name
            stdout: Standard output
            stderr: Standard error
            return_code: Return code
            supports_json: Whether tool supports JSON output

        Returns:
            Parsed LintResult
        """
        output = stdout + "\n" + stderr

        # Try JSON parsing first if supported
        if supports_json and stdout.strip():
            try:
                json_result = self._parse_json_output(tool, stdout)
                if json_result:
                    return json_result
            except:
                pass  # Fall back to text parsing

        # Text parsing by tool
        if tool == "pylint":
            return self._parse_pylint_output(output, return_code)
        elif tool == "flake8":
            return self._parse_flake8_output(output, return_code)
        elif tool == "eslint":
            return self._parse_eslint_text_output(output, return_code)
        elif tool == "rubocop":
            return self._parse_rubocop_text_output(output, return_code)
        elif tool == "rustfmt":
            return self._parse_rustfmt_output(output, return_code)
        else:
            return self._parse_generic_output(tool, output, return_code)

    def _parse_json_output(self, tool: str, json_str: str) -> Optional[LintResult]:
        """Parse JSON output from linters"""
        try:
            data = json.loads(json_str)

            if tool == "eslint":
                return self._parse_eslint_json(data)
            elif tool == "rubocop":
                return self._parse_rubocop_json(data)
            elif tool == "pylint":
                return self._parse_pylint_json(data)

            return None
        except json.JSONDecodeError:
            return None

    def _parse_eslint_json(self, data: List[Dict]) -> LintResult:
        """Parse ESLint JSON output"""
        issues = []
        error_count = 0
        warning_count = 0

        for file_result in data:
            file_path = file_result.get("filePath", "unknown")
            for message in file_result.get("messages", []):
                severity = message.get("severity", 1)
                severity_name = "error" if severity == 2 else "warning"

                if severity == 2:
                    error_count += 1
                else:
                    warning_count += 1

                issues.append(LintIssue(
                    severity=severity_name,
                    message=message.get("message", ""),
                    file_path=file_path,
                    line_number=message.get("line"),
                    column=message.get("column"),
                    rule=message.get("ruleId")
                ))

        status = ValidationStatus.PASSED if error_count == 0 else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool="eslint",
            issues=issues,
            total_issues=len(issues),
            errors=error_count,
            warnings=warning_count,
            output=json.dumps(data)
        )

    def _parse_rubocop_json(self, data: Dict) -> LintResult:
        """Parse Rubocop JSON output"""
        issues = []
        error_count = 0
        warning_count = 0

        for file_result in data.get("files", []):
            file_path = file_result.get("path", "unknown")
            for offense in file_result.get("offenses", []):
                severity = offense.get("severity", "warning")

                if severity in ("error", "fatal"):
                    error_count += 1
                else:
                    warning_count += 1

                location = offense.get("location", {})
                issues.append(LintIssue(
                    severity=severity,
                    message=offense.get("message", ""),
                    file_path=file_path,
                    line_number=location.get("line"),
                    column=location.get("column"),
                    rule=offense.get("cop_name")
                ))

        status = ValidationStatus.PASSED if error_count == 0 else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool="rubocop",
            issues=issues,
            total_issues=len(issues),
            errors=error_count,
            warnings=warning_count,
            output=json.dumps(data)
        )

    def _parse_pylint_json(self, data: List[Dict]) -> LintResult:
        """Parse Pylint JSON output"""
        issues = []
        error_count = 0
        warning_count = 0

        for issue_data in data:
            msg_type = issue_data.get("type", "warning")
            severity = "error" if msg_type in ("error", "fatal") else "warning"

            if severity == "error":
                error_count += 1
            else:
                warning_count += 1

            issues.append(LintIssue(
                severity=severity,
                message=issue_data.get("message", ""),
                file_path=issue_data.get("path", "unknown"),
                line_number=issue_data.get("line"),
                column=issue_data.get("column"),
                rule=issue_data.get("symbol")
            ))

        status = ValidationStatus.PASSED if error_count == 0 else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool="pylint",
            issues=issues,
            total_issues=len(issues),
            errors=error_count,
            warnings=warning_count,
            output=json.dumps(data)
        )

    def _parse_pylint_output(self, output: str, return_code: int) -> LintResult:
        """Parse Pylint text output"""
        # Pylint format: file:line:col: C0111: message (code)
        pattern = r'([^:]+):(\d+):(\d+):\s*([A-Z]\d+):\s*(.+?)\s*\(([^\)]+)\)'

        issues = []
        for line in output.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                code = match.group(4)
                # First letter indicates type: C=convention, R=refactor, W=warning, E=error, F=fatal
                severity = "error" if code[0] in ("E", "F") else "warning"

                issues.append(LintIssue(
                    severity=severity,
                    message=match.group(5),
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                    column=int(match.group(3)),
                    rule=match.group(6)
                ))

        errors = sum(1 for issue in issues if issue.severity == "error")
        warnings = sum(1 for issue in issues if issue.severity == "warning")

        status = ValidationStatus.PASSED if errors == 0 else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool="pylint",
            issues=issues,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            output=output
        )

    def _parse_flake8_output(self, output: str, return_code: int) -> LintResult:
        """Parse Flake8 output"""
        # Flake8 format: file:line:col: code message
        pattern = r'([^:]+):(\d+):(\d+):\s*([A-Z]\d+)\s+(.+)'

        issues = []
        for line in output.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                # Flake8 treats everything as error
                issues.append(LintIssue(
                    severity="warning",  # Flake8 issues are typically warnings
                    message=match.group(5),
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                    column=int(match.group(3)),
                    rule=match.group(4)
                ))

        # Flake8 doesn't distinguish errors/warnings, consider all as warnings
        status = ValidationStatus.PASSED  # Flake8 warnings don't fail validation

        return LintResult(
            status=status,
            tool="flake8",
            issues=issues,
            total_issues=len(issues),
            errors=0,
            warnings=len(issues),
            output=output
        )

    def _parse_eslint_text_output(self, output: str, return_code: int) -> LintResult:
        """Parse ESLint text output (fallback)"""
        # Count errors and warnings from summary
        error_match = re.search(r'(\d+)\s+error', output)
        warning_match = re.search(r'(\d+)\s+warning', output)

        errors = int(error_match.group(1)) if error_match else 0
        warnings = int(warning_match.group(1)) if warning_match else 0

        status = ValidationStatus.PASSED if errors == 0 else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool="eslint",
            issues=[],
            total_issues=errors + warnings,
            errors=errors,
            warnings=warnings,
            output=output
        )

    def _parse_rubocop_text_output(self, output: str, return_code: int) -> LintResult:
        """Parse Rubocop text output (fallback)"""
        # Count offenses from summary
        offense_match = re.search(r'(\d+)\s+offense', output)
        offenses = int(offense_match.group(1)) if offense_match else 0

        status = ValidationStatus.PASSED if offenses == 0 else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool="rubocop",
            issues=[],
            total_issues=offenses,
            errors=offenses,  # Treat as errors
            warnings=0,
            output=output
        )

    def _parse_rustfmt_output(self, output: str, return_code: int) -> LintResult:
        """Parse rustfmt output"""
        # rustfmt outputs differences if format doesn't match
        has_issues = return_code != 0 or "Diff" in output

        status = ValidationStatus.PASSED if not has_issues else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool="rustfmt",
            issues=[],
            total_issues=1 if has_issues else 0,
            errors=1 if has_issues else 0,
            warnings=0,
            output=output
        )

    def _parse_generic_output(self, tool: str, output: str, return_code: int) -> LintResult:
        """Generic parser for unknown linters"""
        # Try to count errors/warnings
        error_count = len(re.findall(r'\berror\b', output, re.IGNORECASE))
        warning_count = len(re.findall(r'\bwarning\b', output, re.IGNORECASE))

        status = ValidationStatus.PASSED if error_count == 0 else ValidationStatus.FAILED

        return LintResult(
            status=status,
            tool=tool,
            issues=[],
            total_issues=error_count + warning_count,
            errors=error_count,
            warnings=warning_count,
            output=output
        )
