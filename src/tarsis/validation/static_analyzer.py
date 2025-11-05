"""
Static analyzer for type checking and static analysis.

Supports mypy, pyright, tsc, flow, and other static analysis tools.
"""

import subprocess
import re
import json
from pathlib import Path
from typing import List, Optional, Dict

from .result_types import AnalysisResult, AnalysisIssue, ValidationStatus


class StaticAnalyzer:
    """
    Runs static analysis and type checking tools.

    Fallback Tier 1 when tests are not available.
    """

    # Tool configurations
    ANALYZERS = {
        "python": [
            {
                "name": "mypy",
                "command": ["mypy", "."],
                "config_files": ["mypy.ini", "setup.cfg", "pyproject.toml"],
            },
            {
                "name": "pyright",
                "command": ["pyright", "."],
                "config_files": ["pyrightconfig.json", "pyproject.toml"],
            },
        ],
        "typescript": [
            {
                "name": "tsc",
                "command": ["tsc", "--noEmit"],
                "config_files": ["tsconfig.json"],
            },
        ],
        "javascript": [
            {
                "name": "flow",
                "command": ["flow", "check"],
                "config_files": [".flowconfig"],
            },
        ],
    }

    def __init__(self, repo_path: str):
        """
        Initialize static analyzer.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)

    async def run_analysis(
        self,
        language: str,
        files: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Run static analysis for given language.

        Args:
            language: Programming language
            files: Optional list of files to analyze

        Returns:
            AnalysisResult with analysis outcomes
        """
        # Get available analyzers for language
        analyzers = self.ANALYZERS.get(language, [])

        if not analyzers:
            return AnalysisResult(
                status=ValidationStatus.SKIPPED,
                tool="none",
                output=f"No static analyzer available for {language}"
            )

        # Try each analyzer until one works
        for analyzer_config in analyzers:
            tool_name = analyzer_config["name"]

            # Check if tool is available
            if not self._is_tool_available(tool_name):
                continue

            # Check if config exists
            has_config = any(
                (self.repo_path / config_file).exists()
                for config_file in analyzer_config["config_files"]
            )

            # Run the analyzer
            try:
                command = analyzer_config["command"].copy()

                # Add files if specified
                if files and tool_name in ("mypy", "pyright"):
                    # Replace "." with specific files
                    if "." in command:
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
                analysis_result = self._parse_analyzer_output(
                    tool_name,
                    result.stdout,
                    result.stderr,
                    result.returncode
                )

                return analysis_result

            except subprocess.TimeoutExpired:
                return AnalysisResult(
                    status=ValidationStatus.ERROR,
                    tool=tool_name,
                    error_message="Static analysis timed out (3 minutes)"
                )
            except FileNotFoundError:
                # Tool not found, try next
                continue
            except Exception as e:
                return AnalysisResult(
                    status=ValidationStatus.ERROR,
                    tool=tool_name,
                    error_message=f"Failed to run {tool_name}: {str(e)}"
                )

        # No analyzer worked
        return AnalysisResult(
            status=ValidationStatus.SKIPPED,
            tool="none",
            output=f"No working static analyzer found for {language}"
        )

    def _is_tool_available(self, tool_name: str) -> bool:
        """Check if tool is available in PATH"""
        try:
            subprocess.run(
                [tool_name, "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _parse_analyzer_output(
        self,
        tool: str,
        stdout: str,
        stderr: str,
        return_code: int
    ) -> AnalysisResult:
        """
        Parse output from static analyzer.

        Args:
            tool: Tool name
            stdout: Standard output
            stderr: Standard error
            return_code: Return code

        Returns:
            Parsed AnalysisResult
        """
        output = stdout + "\n" + stderr

        if tool == "mypy":
            return self._parse_mypy_output(output, return_code)
        elif tool == "pyright":
            return self._parse_pyright_output(output, return_code)
        elif tool == "tsc":
            return self._parse_tsc_output(output, return_code)
        elif tool == "flow":
            return self._parse_flow_output(output, return_code)
        else:
            return self._parse_generic_output(tool, output, return_code)

    def _parse_mypy_output(self, output: str, return_code: int) -> AnalysisResult:
        """Parse mypy output"""
        # mypy format: file:line:col: error: message [code]
        pattern = r'([^:]+):(\d+):(\d+):\s*(error|warning|note):\s*(.+?)(?:\s+\[([^\]]+)\])?$'

        issues = []
        for line in output.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                file_path = match.group(1)
                line_num = int(match.group(2))
                col = int(match.group(3))
                severity = match.group(4)
                message = match.group(5)
                code = match.group(6) if match.group(6) else None

                issues.append(AnalysisIssue(
                    severity=severity,
                    message=message,
                    file_path=file_path,
                    line_number=line_num,
                    column=col,
                    code=code
                ))

        # Count by severity
        errors = sum(1 for issue in issues if issue.severity == "error")
        warnings = sum(1 for issue in issues if issue.severity == "warning")

        # Determine status
        if return_code == 0:
            status = ValidationStatus.PASSED
        elif errors > 0:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED  # Only warnings

        return AnalysisResult(
            status=status,
            tool="mypy",
            issues=issues,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            output=output
        )

    def _parse_pyright_output(self, output: str, return_code: int) -> AnalysisResult:
        """Parse pyright output"""
        # Try JSON output first
        try:
            # Pyright can output JSON with --outputjson flag
            # For now, parse text output
            pass
        except:
            pass

        # Text format: file:line:col - error/warning: message
        pattern = r'([^:]+):(\d+):(\d+)\s+-\s+(error|warning|information):\s+(.+?)(?:\s+\(([^\)]+)\))?$'

        issues = []
        for line in output.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                issues.append(AnalysisIssue(
                    severity=match.group(4),
                    message=match.group(5),
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                    column=int(match.group(3)),
                    code=match.group(6) if match.group(6) else None
                ))

        errors = sum(1 for issue in issues if issue.severity == "error")
        warnings = sum(1 for issue in issues if issue.severity == "warning")

        status = ValidationStatus.PASSED if return_code == 0 and errors == 0 else ValidationStatus.FAILED

        return AnalysisResult(
            status=status,
            tool="pyright",
            issues=issues,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            output=output
        )

    def _parse_tsc_output(self, output: str, return_code: int) -> AnalysisResult:
        """Parse TypeScript compiler output"""
        # tsc format: file(line,col): error TS####: message
        pattern = r'([^(]+)\((\d+),(\d+)\):\s+(error|warning)\s+TS(\d+):\s+(.+)$'

        issues = []
        for line in output.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                issues.append(AnalysisIssue(
                    severity=match.group(4),
                    message=match.group(6),
                    file_path=match.group(1),
                    line_number=int(match.group(2)),
                    column=int(match.group(3)),
                    code=f"TS{match.group(5)}"
                ))

        errors = sum(1 for issue in issues if issue.severity == "error")
        warnings = sum(1 for issue in issues if issue.severity == "warning")

        status = ValidationStatus.PASSED if return_code == 0 and errors == 0 else ValidationStatus.FAILED

        return AnalysisResult(
            status=status,
            tool="tsc",
            issues=issues,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            output=output
        )

    def _parse_flow_output(self, output: str, return_code: int) -> AnalysisResult:
        """Parse Flow output"""
        # Flow can output JSON
        try:
            # Try JSON parsing
            if output.strip().startswith('{'):
                data = json.loads(output)
                if "errors" in data:
                    issues = []
                    for error in data["errors"]:
                        # Flow errors are complex nested structures
                        message = error.get("message", [{}])[0].get("descr", "Error")
                        issues.append(AnalysisIssue(
                            severity="error",
                            message=message,
                            file_path="unknown"
                        ))

                    return AnalysisResult(
                        status=ValidationStatus.FAILED if issues else ValidationStatus.PASSED,
                        tool="flow",
                        issues=issues,
                        total_issues=len(issues),
                        errors=len(issues),
                        warnings=0,
                        output=output
                    )
        except json.JSONDecodeError:
            pass

        # Text parsing fallback
        status = ValidationStatus.PASSED if return_code == 0 else ValidationStatus.FAILED

        return AnalysisResult(
            status=status,
            tool="flow",
            issues=[],
            total_issues=0,
            errors=0,
            warnings=0,
            output=output
        )

    def _parse_generic_output(self, tool: str, output: str, return_code: int) -> AnalysisResult:
        """Generic parser for unknown tools"""
        # Try to find error/warning patterns
        error_count = len(re.findall(r'\berror\b', output, re.IGNORECASE))
        warning_count = len(re.findall(r'\bwarning\b', output, re.IGNORECASE))

        status = ValidationStatus.PASSED if return_code == 0 and error_count == 0 else ValidationStatus.FAILED

        return AnalysisResult(
            status=status,
            tool=tool,
            issues=[],
            total_issues=error_count + warning_count,
            errors=error_count,
            warnings=warning_count,
            output=output
        )
