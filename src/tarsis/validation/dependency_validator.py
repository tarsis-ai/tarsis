"""
Dependency and import validator.

Checks that imports are valid and dependencies are properly declared.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import List, Optional, Set, Dict
import ast

from .result_types import ValidationStatus


class DependencyIssue:
    """Issue found during dependency validation"""
    def __init__(
        self,
        severity: str,
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        dependency: Optional[str] = None
    ):
        self.severity = severity
        self.message = message
        self.file_path = file_path
        self.line_number = line_number
        self.dependency = dependency


class DependencyResult:
    """Result of dependency validation"""
    def __init__(
        self,
        status: ValidationStatus,
        issues: List[DependencyIssue] = None,
        total_issues: int = 0,
        errors: int = 0,
        warnings: int = 0,
        output: str = "",
        error_message: Optional[str] = None
    ):
        self.status = status
        self.issues = issues or []
        self.total_issues = total_issues
        self.errors = errors
        self.warnings = warnings
        self.output = output
        self.error_message = error_message

    @property
    def passed(self) -> bool:
        """Check if validation passed (no errors)"""
        return self.status == ValidationStatus.PASSED and self.errors == 0


class DependencyValidator:
    """
    Validates imports and dependencies across multiple languages.

    Checks:
    - Invalid imports
    - Missing dependencies
    - Unused dependencies (optional)
    """

    def __init__(self, repo_path: str):
        """
        Initialize dependency validator.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)

    async def validate_dependencies(
        self,
        language: str,
        files: Optional[List[str]] = None
    ) -> DependencyResult:
        """
        Validate dependencies for given language.

        Args:
            language: Programming language
            files: Optional list of files to check

        Returns:
            DependencyResult with validation outcomes
        """
        try:
            if language == "python":
                return await self._validate_python_dependencies(files)
            elif language in ("javascript", "typescript"):
                return await self._validate_node_dependencies(files)
            elif language == "go":
                return await self._validate_go_dependencies(files)
            elif language == "rust":
                return await self._validate_rust_dependencies(files)
            else:
                return DependencyResult(
                    status=ValidationStatus.SKIPPED,
                    output=f"Dependency validation not supported for {language}"
                )

        except Exception as e:
            return DependencyResult(
                status=ValidationStatus.ERROR,
                error_message=f"Failed to validate dependencies: {str(e)}",
                output=str(e)
            )

    async def _validate_python_dependencies(
        self,
        files: Optional[List[str]] = None
    ) -> DependencyResult:
        """Validate Python imports and dependencies"""
        issues = []

        # Get imports from Python files
        imports = self._extract_python_imports(files)

        # Check if imports are valid (can be imported)
        for file_path, file_imports in imports.items():
            for imp in file_imports:
                if not self._can_import_python(imp):
                    issues.append(DependencyIssue(
                        severity="error",
                        message=f"Cannot import '{imp}' - module not found",
                        file_path=file_path,
                        dependency=imp
                    ))

        # Check requirements.txt if exists
        req_issues = self._check_python_requirements()
        issues.extend(req_issues)

        # Determine status
        errors = sum(1 for issue in issues if issue.severity == "error")
        warnings = sum(1 for issue in issues if issue.severity == "warning")

        if errors > 0:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return DependencyResult(
            status=status,
            issues=issues,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            output=f"Checked Python dependencies: {errors} error(s), {warnings} warning(s)"
        )

    async def _validate_node_dependencies(
        self,
        files: Optional[List[str]] = None
    ) -> DependencyResult:
        """Validate Node.js imports and dependencies"""
        issues = []

        # Check package.json exists
        package_json = self.repo_path / "package.json"
        if not package_json.exists():
            return DependencyResult(
                status=ValidationStatus.SKIPPED,
                output="No package.json found"
            )

        # Load package.json
        try:
            with open(package_json) as f:
                package_data = json.load(f)
        except Exception as e:
            return DependencyResult(
                status=ValidationStatus.ERROR,
                error_message=f"Failed to parse package.json: {str(e)}"
            )

        # Get declared dependencies
        dependencies = set(package_data.get("dependencies", {}).keys())
        dev_dependencies = set(package_data.get("devDependencies", {}).keys())
        all_deps = dependencies | dev_dependencies

        # Check if node_modules exists
        node_modules = self.repo_path / "node_modules"
        if not node_modules.exists():
            issues.append(DependencyIssue(
                severity="warning",
                message="node_modules not found - run 'npm install'",
            ))

        # Extract imports from JS/TS files
        imports = self._extract_node_imports(files)

        # Check if imported modules are in dependencies
        for file_path, file_imports in imports.items():
            for imp in file_imports:
                # Skip relative imports
                if imp.startswith('.') or imp.startswith('/'):
                    continue

                # Extract package name (handle scoped packages)
                package_name = self._extract_package_name(imp)

                # Check if it's a built-in module
                if self._is_node_builtin(package_name):
                    continue

                # Check if it's in dependencies
                if package_name not in all_deps:
                    issues.append(DependencyIssue(
                        severity="error",
                        message=f"Import '{imp}' not found in package.json dependencies",
                        file_path=file_path,
                        dependency=package_name
                    ))

        errors = sum(1 for issue in issues if issue.severity == "error")
        warnings = sum(1 for issue in issues if issue.severity == "warning")

        status = ValidationStatus.PASSED if errors == 0 else ValidationStatus.FAILED

        return DependencyResult(
            status=status,
            issues=issues,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            output=f"Checked Node.js dependencies: {errors} error(s), {warnings} warning(s)"
        )

    async def _validate_go_dependencies(
        self,
        files: Optional[List[str]] = None
    ) -> DependencyResult:
        """Validate Go imports"""
        # Check if go.mod exists
        go_mod = self.repo_path / "go.mod"
        if not go_mod.exists():
            return DependencyResult(
                status=ValidationStatus.SKIPPED,
                output="No go.mod found"
            )

        # Run go mod verify
        try:
            result = subprocess.run(
                ["go", "mod", "verify"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.repo_path)
            )

            if result.returncode == 0:
                return DependencyResult(
                    status=ValidationStatus.PASSED,
                    output="Go modules verified successfully"
                )
            else:
                return DependencyResult(
                    status=ValidationStatus.FAILED,
                    errors=1,
                    total_issues=1,
                    output=result.stderr or result.stdout
                )

        except FileNotFoundError:
            return DependencyResult(
                status=ValidationStatus.SKIPPED,
                output="Go not installed"
            )
        except Exception as e:
            return DependencyResult(
                status=ValidationStatus.ERROR,
                error_message=str(e)
            )

    async def _validate_rust_dependencies(
        self,
        files: Optional[List[str]] = None
    ) -> DependencyResult:
        """Validate Rust dependencies"""
        # Check if Cargo.toml exists
        cargo_toml = self.repo_path / "Cargo.toml"
        if not cargo_toml.exists():
            return DependencyResult(
                status=ValidationStatus.SKIPPED,
                output="No Cargo.toml found"
            )

        # Run cargo check (light check without building)
        try:
            result = subprocess.run(
                ["cargo", "check", "--message-format=short"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.repo_path)
            )

            # Parse output for dependency issues
            issues = []
            for line in result.stderr.split('\n'):
                if 'error' in line.lower() and 'dependency' in line.lower():
                    issues.append(DependencyIssue(
                        severity="error",
                        message=line.strip()
                    ))

            if result.returncode == 0:
                return DependencyResult(
                    status=ValidationStatus.PASSED,
                    output="Cargo dependencies validated successfully"
                )
            else:
                return DependencyResult(
                    status=ValidationStatus.FAILED,
                    issues=issues,
                    errors=len(issues),
                    total_issues=len(issues),
                    output=result.stderr
                )

        except FileNotFoundError:
            return DependencyResult(
                status=ValidationStatus.SKIPPED,
                output="Cargo not installed"
            )
        except Exception as e:
            return DependencyResult(
                status=ValidationStatus.ERROR,
                error_message=str(e)
            )

    def _extract_python_imports(
        self,
        files: Optional[List[str]] = None
    ) -> Dict[str, Set[str]]:
        """Extract imports from Python files"""
        imports = {}

        # Get Python files
        if files is None:
            py_files = list(self.repo_path.rglob("*.py"))
            files = [str(f.relative_to(self.repo_path)) for f in py_files[:50]]  # Limit

        for file_path in files:
            full_path = self.repo_path / file_path
            if not full_path.exists() or not str(file_path).endswith('.py'):
                continue

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(file_path))

                file_imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            file_imports.add(node.module.split('.')[0])

                if file_imports:
                    imports[file_path] = file_imports

            except Exception:
                # Skip files that can't be parsed
                pass

        return imports

    def _can_import_python(self, module_name: str) -> bool:
        """Check if a Python module can be imported"""
        # Skip standard library and common built-ins
        stdlib = {
            'os', 'sys', 'time', 'datetime', 'json', 're', 'math', 'random',
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'subprocess', 'argparse', 'logging', 'unittest', 'asyncio',
            'dataclasses', 'enum', 'abc', 'io', 'tempfile', 'shutil'
        }

        if module_name in stdlib:
            return True

        # Try to import
        try:
            __import__(module_name)
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def _check_python_requirements(self) -> List[DependencyIssue]:
        """Check Python requirements.txt"""
        issues = []

        req_file = self.repo_path / "requirements.txt"
        if not req_file.exists():
            return issues

        try:
            with open(req_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Extract package name (before ==, >=, etc.)
                    package = re.split(r'[=<>!]', line)[0].strip()

                    if not self._can_import_python(package):
                        issues.append(DependencyIssue(
                            severity="warning",
                            message=f"Package '{package}' in requirements.txt not installed",
                            file_path="requirements.txt",
                            line_number=line_num,
                            dependency=package
                        ))
        except Exception:
            pass

        return issues

    def _extract_node_imports(
        self,
        files: Optional[List[str]] = None
    ) -> Dict[str, Set[str]]:
        """Extract imports from JavaScript/TypeScript files"""
        imports = {}

        # Get JS/TS files
        if files is None:
            patterns = ["*.js", "*.jsx", "*.ts", "*.tsx"]
            js_files = []
            for pattern in patterns:
                js_files.extend(list(self.repo_path.rglob(pattern)))
            files = [str(f.relative_to(self.repo_path)) for f in js_files[:50]]

        for file_path in files:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                continue

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                file_imports = set()

                # Match import statements
                # import foo from 'package'
                # import { bar } from 'package'
                import_pattern = r'import\s+(?:[\w{},\s]+\s+from\s+)?[\'"]([^\'"]+)[\'"]'
                for match in re.finditer(import_pattern, content):
                    file_imports.add(match.group(1))

                # Match require statements
                # const foo = require('package')
                require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
                for match in re.finditer(require_pattern, content):
                    file_imports.add(match.group(1))

                if file_imports:
                    imports[file_path] = file_imports

            except Exception:
                pass

        return imports

    def _extract_package_name(self, import_path: str) -> str:
        """Extract package name from import path"""
        # Handle scoped packages (@scope/package)
        if import_path.startswith('@'):
            parts = import_path.split('/')
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            return parts[0]

        # Regular package
        return import_path.split('/')[0]

    def _is_node_builtin(self, package_name: str) -> bool:
        """Check if package is a Node.js built-in module"""
        builtins = {
            'fs', 'path', 'http', 'https', 'url', 'querystring', 'os', 'util',
            'events', 'stream', 'buffer', 'crypto', 'zlib', 'assert', 'child_process',
            'cluster', 'dgram', 'dns', 'domain', 'net', 'readline', 'repl',
            'tls', 'tty', 'vm', 'process', 'console', 'timers', 'module'
        }
        return package_name in builtins
