"""
Validation result reporter.

Formats validation results for different outputs: PR comments, status badges, and agent feedback.
"""

from typing import Optional
from .result_types import (
    ValidationResult,
    ValidationStatus,
    TestResult,
    AnalysisResult,
    LintResult,
    SyntaxResult
)
from .detector import ValidationTier


class ValidationReporter:
    """
    Formats validation results for various outputs.
    """

    @staticmethod
    def generate_status_badge(result: ValidationResult) -> str:
        """
        Generate a status badge/icon for the validation result.

        Args:
            result: Validation result

        Returns:
            Badge text (emoji + status)
        """
        if result.status == ValidationStatus.PASSED:
            return "✅ PASSED"
        elif result.status == ValidationStatus.FAILED:
            return "❌ FAILED"
        elif result.status == ValidationStatus.SKIPPED:
            return "⏭️ SKIPPED"
        elif result.status == ValidationStatus.ERROR:
            return "⚠️ ERROR"
        return "❓ UNKNOWN"

    @staticmethod
    def generate_pr_comment(result: ValidationResult) -> str:
        """
        Generate a formatted PR comment with validation results.

        Args:
            result: Validation result

        Returns:
            Markdown-formatted comment text
        """
        badge = ValidationReporter.generate_status_badge(result)
        tier_name = result.tier_used.value.replace("_", " ").title()

        comment = f"""## {badge} Validation Results

**Validation Tier:** {tier_name}
"""

        # Add user decision if present
        if result.user_decision:
            comment += f"**User Decision:** {result.user_decision}\n"

        comment += f"**Duration:** {result.duration:.2f}s\n\n"

        # Add summary
        if result.summary:
            comment += f"### Summary\n{result.summary}\n\n"

        # Add tier-specific results
        if result.test_result:
            comment += ValidationReporter._format_test_result(result.test_result)
        elif result.analysis_result:
            comment += ValidationReporter._format_analysis_result(result.analysis_result)
        elif result.lint_result:
            comment += ValidationReporter._format_lint_result(result.lint_result)
        elif result.syntax_result:
            comment += ValidationReporter._format_syntax_result(result.syntax_result)

        # Add details if present
        if result.details:
            comment += f"\n### Details\n{result.details}\n"

        # Add footer
        comment += "\n---\n*Automated validation by Tarsis*\n"

        return comment

    @staticmethod
    def _format_test_result(test_result: TestResult) -> str:
        """Format test result section"""
        text = "### Test Results\n\n"

        if test_result.status == ValidationStatus.ERROR:
            text += f"⚠️ **Error running tests:** {test_result.error_message}\n"
            return text

        # Stats
        text += f"- **Total:** {test_result.total_tests}\n"
        text += f"- **Passed:** {test_result.passed_tests} ✅\n"
        text += f"- **Failed:** {test_result.failed_tests} ❌\n"

        if test_result.skipped_tests > 0:
            text += f"- **Skipped:** {test_result.skipped_tests} ⏭️\n"

        text += f"- **Duration:** {test_result.duration:.2f}s\n\n"

        # Show failures if any
        if test_result.failures:
            text += "#### Failures\n\n"
            for i, failure in enumerate(test_result.failures[:10], 1):  # Limit to 10
                text += f"{i}. **{failure.test_name}**\n"
                if failure.file_path:
                    text += f"   - File: `{failure.file_path}`"
                    if failure.line_number:
                        text += f":{failure.line_number}"
                    text += "\n"
                text += f"   - Error: {failure.error_message}\n\n"

            if len(test_result.failures) > 10:
                text += f"*... and {len(test_result.failures) - 10} more failures*\n\n"

        return text

    @staticmethod
    def _format_analysis_result(analysis_result: AnalysisResult) -> str:
        """Format static analysis result section"""
        text = f"### Static Analysis ({analysis_result.tool})\n\n"

        if analysis_result.status == ValidationStatus.ERROR:
            text += f"⚠️ **Error running analysis:** {analysis_result.error_message}\n"
            return text

        # Stats
        text += f"- **Total Issues:** {analysis_result.total_issues}\n"
        text += f"- **Errors:** {analysis_result.errors} ❌\n"
        text += f"- **Warnings:** {analysis_result.warnings} ⚠️\n\n"

        # Show issues if any
        if analysis_result.issues:
            text += "#### Issues\n\n"
            for i, issue in enumerate(analysis_result.issues[:15], 1):  # Limit to 15
                severity_icon = "❌" if issue.severity == "error" else "⚠️"
                text += f"{i}. {severity_icon} `{issue.file_path}`"
                if issue.line_number:
                    text += f":{issue.line_number}"
                    if issue.column:
                        text += f":{issue.column}"
                text += "\n"
                text += f"   - {issue.message}\n"
                if issue.code:
                    text += f"   - Code: `{issue.code}`\n"
                text += "\n"

            if len(analysis_result.issues) > 15:
                text += f"*... and {len(analysis_result.issues) - 15} more issues*\n\n"

        return text

    @staticmethod
    def _format_lint_result(lint_result: LintResult) -> str:
        """Format linting result section"""
        text = f"### Linting ({lint_result.tool})\n\n"

        if lint_result.status == ValidationStatus.ERROR:
            text += f"⚠️ **Error running linter:** {lint_result.error_message}\n"
            return text

        # Stats
        text += f"- **Total Issues:** {lint_result.total_issues}\n"
        text += f"- **Errors:** {lint_result.errors} ❌\n"
        text += f"- **Warnings:** {lint_result.warnings} ⚠️\n\n"

        # Show issues if any
        if lint_result.issues:
            text += "#### Issues\n\n"
            for i, issue in enumerate(lint_result.issues[:15], 1):  # Limit to 15
                severity_icon = "❌" if issue.severity == "error" else "⚠️"
                text += f"{i}. {severity_icon} `{issue.file_path}`"
                if issue.line_number:
                    text += f":{issue.line_number}"
                text += "\n"
                text += f"   - {issue.message}\n"
                if issue.rule:
                    text += f"   - Rule: `{issue.rule}`\n"
                text += "\n"

            if len(lint_result.issues) > 15:
                text += f"*... and {len(lint_result.issues) - 15} more issues*\n\n"

        return text

    @staticmethod
    def _format_syntax_result(syntax_result: SyntaxResult) -> str:
        """Format syntax checking result section"""
        text = "### Syntax Checking\n\n"

        if syntax_result.status == ValidationStatus.ERROR:
            text += f"⚠️ **Error checking syntax:** {syntax_result.error_message}\n"
            return text

        # Stats
        text += f"- **Files Checked:** {syntax_result.files_checked}\n"
        text += f"- **Errors:** {syntax_result.total_errors} ❌\n\n"

        # Show errors if any
        if syntax_result.errors:
            text += "#### Syntax Errors\n\n"
            for i, error in enumerate(syntax_result.errors[:10], 1):  # Limit to 10
                text += f"{i}. ❌ `{error.file_path}`"
                if error.line_number:
                    text += f":{error.line_number}"
                    if error.column:
                        text += f":{error.column}"
                text += "\n"
                text += f"   - {error.message}\n\n"

            if len(syntax_result.errors) > 10:
                text += f"*... and {len(syntax_result.errors) - 10} more errors*\n\n"

        return text

    @staticmethod
    def format_for_agent(result: ValidationResult) -> str:
        """
        Format validation result for agent consumption (concise feedback).

        Args:
            result: Validation result

        Returns:
            Concise text summary for the agent
        """
        badge = ValidationReporter.generate_status_badge(result)
        tier_name = result.tier_used.value.replace("_", " ").title()

        text = f"{badge}\n\n"
        text += f"**Validation Tier:** {tier_name}\n"
        text += f"**Duration:** {result.duration:.2f}s\n\n"

        # Add quick summary
        if result.summary:
            text += f"{result.summary}\n\n"

        # Add failure details if failed
        if result.has_failures:
            text += f"**Failures:** {result.get_failure_summary()}\n\n"

            # Add specific failure info
            if result.test_result and result.test_result.failures:
                text += "**Failed Tests:**\n"
                for failure in result.test_result.failures[:5]:
                    text += f"- {failure.test_name}: {failure.error_message}\n"
                if len(result.test_result.failures) > 5:
                    text += f"  ... and {len(result.test_result.failures) - 5} more\n"

        return text

    @staticmethod
    def format_summary_line(result: ValidationResult) -> str:
        """
        Generate a one-line summary of validation results.

        Args:
            result: Validation result

        Returns:
            One-line summary
        """
        badge = ValidationReporter.generate_status_badge(result)
        tier_name = result.tier_used.value.replace("_", " ").title()

        if result.passed:
            return f"{badge} ({tier_name})"
        else:
            return f"{badge} ({tier_name}): {result.get_failure_summary()}"
