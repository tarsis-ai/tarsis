"""
Syntax checker for validating code syntax across multiple languages.

Provides basic syntax validation as a fallback when no tests or higher-tier
validation is available.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
import re

from .result_types import SyntaxResult, SyntaxError as SyntaxErr, ValidationStatus


class SyntaxChecker:
    """
    Checks syntax of code files across multiple programming languages.

    Always available fallback validation (Tier 4).
    """

    # Language-specific syntax checkers
    SYNTAX_CHECKERS = {
        "python": {
            "command": ["python", "-m", "py_compile"],
            "file_arg": True,  # File path is passed as argument
        },
        "javascript": {
            "command": ["node", "--check"],
            "file_arg": True,
        },
        "typescript": {
            "command": ["node", "--check"],  # Fallback to node if tsc not available
            "file_arg": True,
        },
        "go": {
            "command": ["gofmt", "-e"],
            "file_arg": True,
        },
        "rust": {
            "command": ["rustc", "--crate-type", "lib", "-Z", "parse-only"],
            "file_arg": True,
        },
        "ruby": {
            "command": ["ruby", "-c"],
            "file_arg": True,
        },
        "java": {
            "command": ["javac", "-Xstdout"],
            "file_arg": True,
        }
    }

    def __init__(self, repo_path: str):
        """
        Initialize syntax checker.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)

    async def check_syntax(
        self,
        files: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> SyntaxResult:
        """
        Check syntax of files.

        Args:
            files: Optional list of files to check (relative to repo_path)
                  If None, checks all relevant files
            language: Optional language hint (auto-detected if None)

        Returns:
            SyntaxResult with syntax check outcomes
        """
        try:
            # Determine files to check
            if files is None:
                files = self._get_all_source_files(language)

            if not files:
                return SyntaxResult(
                    status=ValidationStatus.SKIPPED,
                    files_checked=0,
                    output="No files to check"
                )

            # Check each file
            errors = []
            files_checked = 0

            for file_path in files:
                file_errors = await self._check_file_syntax(file_path, language)
                errors.extend(file_errors)
                files_checked += 1

            # Determine status
            if errors:
                status = ValidationStatus.FAILED
            else:
                status = ValidationStatus.PASSED

            return SyntaxResult(
                status=status,
                errors=errors,
                total_errors=len(errors),
                files_checked=files_checked,
                output=f"Checked {files_checked} files, found {len(errors)} syntax errors"
            )

        except Exception as e:
            return SyntaxResult(
                status=ValidationStatus.ERROR,
                error_message=f"Failed to check syntax: {str(e)}",
                output=str(e)
            )

    async def _check_file_syntax(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> List[SyntaxErr]:
        """
        Check syntax of a single file.

        Args:
            file_path: File path relative to repo_path
            language: Optional language hint

        Returns:
            List of syntax errors found
        """
        full_path = self.repo_path / file_path

        if not full_path.exists():
            return [SyntaxErr(
                file_path=file_path,
                message=f"File not found: {file_path}"
            )]

        # Detect language if not provided
        if language is None:
            language = self._detect_language(full_path)

        if language is None:
            # Unknown language, skip
            return []

        # Get syntax checker for language
        checker_config = self.SYNTAX_CHECKERS.get(language)
        if checker_config is None:
            # No syntax checker for this language
            return []

        # Run syntax check
        try:
            command = checker_config["command"].copy()
            if checker_config["file_arg"]:
                command.append(str(full_path))

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.repo_path)
            )

            # Parse errors from output
            if result.returncode != 0:
                errors = self._parse_syntax_errors(
                    file_path,
                    result.stderr or result.stdout,
                    language
                )
                return errors

            return []

        except subprocess.TimeoutExpired:
            return [SyntaxErr(
                file_path=file_path,
                message="Syntax check timed out"
            )]
        except FileNotFoundError:
            # Checker command not found, try alternative
            alt_errors = await self._try_alternative_checker(file_path, language)
            return alt_errors
        except Exception as e:
            return [SyntaxErr(
                file_path=file_path,
                message=f"Error checking syntax: {str(e)}"
            )]

    async def _try_alternative_checker(
        self,
        file_path: str,
        language: str
    ) -> List[SyntaxErr]:
        """
        Try alternative syntax checking method.

        For some languages, we can try to import/compile as a fallback.

        Args:
            file_path: File path
            language: Language

        Returns:
            List of syntax errors
        """
        full_path = self.repo_path / file_path

        # Language-specific alternatives
        if language == "python":
            # Try to compile with Python
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, file_path, 'exec')
                return []
            except SyntaxError as e:
                return [SyntaxErr(
                    file_path=file_path,
                    line_number=e.lineno,
                    column=e.offset,
                    message=str(e.msg)
                )]
            except Exception as e:
                return [SyntaxErr(
                    file_path=file_path,
                    message=f"Error: {str(e)}"
                )]

        # For other languages, just skip if checker not available
        return []

    def _parse_syntax_errors(
        self,
        file_path: str,
        error_output: str,
        language: str
    ) -> List[SyntaxErr]:
        """
        Parse syntax errors from checker output.

        Args:
            file_path: File path
            error_output: Error output from checker
            language: Language

        Returns:
            List of parsed syntax errors
        """
        errors = []

        # Language-specific parsing
        if language == "python":
            # Python error format: File "...", line X, ...
            pattern = r'File "([^"]+)", line (\d+)'
            for line in error_output.split('\n'):
                match = re.search(pattern, line)
                if match:
                    line_num = int(match.group(2))
                    # Get the error message (usually next line or same line)
                    msg_match = re.search(r'(SyntaxError|IndentationError|TabError):\s*(.+)', error_output)
                    message = msg_match.group(2) if msg_match else line.strip()

                    errors.append(SyntaxErr(
                        file_path=file_path,
                        line_number=line_num,
                        message=message
                    ))

        elif language in ("javascript", "typescript"):
            # Node error format: file:line:column - error
            pattern = r':(\d+):(\d+)'
            for line in error_output.split('\n'):
                if file_path in line or 'SyntaxError' in line:
                    match = re.search(pattern, line)
                    if match:
                        errors.append(SyntaxErr(
                            file_path=file_path,
                            line_number=int(match.group(1)),
                            column=int(match.group(2)),
                            message=line.strip()
                        ))

        elif language == "go":
            # Go error format: file:line:col: message
            pattern = r'([^:]+):(\d+):(\d+):\s*(.+)'
            for line in error_output.split('\n'):
                match = re.search(pattern, line)
                if match:
                    errors.append(SyntaxErr(
                        file_path=file_path,
                        line_number=int(match.group(2)),
                        column=int(match.group(3)),
                        message=match.group(4)
                    ))

        elif language == "rust":
            # Rust error format: error: message --> file:line:col
            pattern = r'-->\s*([^:]+):(\d+):(\d+)'
            for line in error_output.split('\n'):
                match = re.search(pattern, line)
                if match:
                    errors.append(SyntaxErr(
                        file_path=file_path,
                        line_number=int(match.group(2)),
                        column=int(match.group(3)),
                        message="Syntax error"
                    ))

        # If no errors parsed but output exists, create generic error
        if not errors and error_output.strip():
            errors.append(SyntaxErr(
                file_path=file_path,
                message=error_output.strip()[:200]  # Limit message length
            ))

        return errors

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """
        Detect language from file extension.

        Args:
            file_path: File path

        Returns:
            Language name or None
        """
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".java": "java",
        }

        return ext_to_lang.get(file_path.suffix.lower())

    def _get_all_source_files(self, language: Optional[str] = None) -> List[str]:
        """
        Get all source files in repository.

        Args:
            language: Optional language filter

        Returns:
            List of file paths relative to repo_path
        """
        files = []

        # Extensions to search for
        if language:
            exts = self._get_extensions_for_language(language)
        else:
            exts = [".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".rb", ".java"]

        for ext in exts:
            for file_path in self.repo_path.rglob(f"*{ext}"):
                if self._should_check_file(file_path):
                    rel_path = file_path.relative_to(self.repo_path)
                    files.append(str(rel_path))

        return files[:100]  # Limit to 100 files

    def _get_extensions_for_language(self, language: str) -> List[str]:
        """Get file extensions for a language"""
        lang_to_exts = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "go": [".go"],
            "rust": [".rs"],
            "ruby": [".rb"],
            "java": [".java"],
        }
        return lang_to_exts.get(language, [])

    def _should_check_file(self, file_path: Path) -> bool:
        """Check if file should be checked (exclude build dirs, etc.)"""
        exclude_patterns = [
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".egg-info", "target", ".pytest_cache",
            "vendor", ".tox"
        ]

        path_str = str(file_path)
        return not any(pattern in path_str for pattern in exclude_patterns)
