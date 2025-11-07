"""
Test framework and validation tier detection.

Detects available testing frameworks and validation tools in a repository.
"""

import os
import subprocess
from typing import List, Optional, Set, Dict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ValidationTier(Enum):
    """Available validation tiers in order of preference"""
    TESTS = "tests"
    STATIC_ANALYSIS = "static_analysis"
    LINTING = "linting"
    SYNTAX = "syntax"


@dataclass
class TestDetectionResult:
    """Result of test framework detection"""
    has_tests: bool
    framework: Optional[str] = None  # pytest, jest, unittest, etc.
    test_files: List[str] = field(default_factory=list)
    test_directories: List[str] = field(default_factory=list)
    test_command: Optional[str] = None
    config_files: List[str] = field(default_factory=list)
    validation_tier: ValidationTier = ValidationTier.SYNTAX
    available_tiers: List[ValidationTier] = field(default_factory=list)
    language: Optional[str] = None  # python, javascript, go, etc.


class TestFrameworkDetector:
    """
    Detects test frameworks across multiple languages.
    """

    # Test directory patterns by language
    TEST_DIRECTORIES = {
        "python": ["tests", "test", "__tests__", "testing"],
        "javascript": ["test", "tests", "__tests__", "spec", "__test__"],
        "typescript": ["test", "tests", "__tests__", "spec"],
        "go": ["_test"],
        "rust": ["tests"],
        "java": ["test", "tests", "src/test"],
        "ruby": ["test", "spec"],
    }

    # Test file patterns by language
    TEST_FILE_PATTERNS = {
        "python": ["test_*.py", "*_test.py", "tests.py"],
        "javascript": ["*.test.js", "*.spec.js", "*Test.js", "*_test.js"],
        "typescript": ["*.test.ts", "*.spec.ts", "*.test.tsx", "*.spec.tsx"],
        "go": ["*_test.go"],
        "rust": ["tests.rs", "test_*.rs"],
        "java": ["*Test.java", "*Tests.java"],
        "ruby": ["*_spec.rb", "*_test.rb"],
    }

    # Framework config files
    FRAMEWORK_CONFIGS = {
        "pytest": ["pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"],
        "unittest": ["setup.py"],
        "jest": ["jest.config.js", "jest.config.ts", "jest.config.json", "package.json"],
        "mocha": [".mocharc.json", ".mocharc.js", "package.json"],
        "vitest": ["vitest.config.ts", "vitest.config.js"],
        "go_test": ["go.mod"],
        "cargo_test": ["Cargo.toml"],
        "rspec": [".rspec", "spec/spec_helper.rb"],
    }

    def __init__(self, repo_path: str):
        """
        Initialize detector.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)

    def detect(self) -> TestDetectionResult:
        """
        Detect test framework and available validation tiers.

        Returns:
            TestDetectionResult with detection information
        """
        # Detect primary language
        language = self._detect_language()

        # Find test directories
        test_dirs = self._find_test_directories(language)

        # Find test files
        test_files = self._find_test_files(language)

        # Detect framework
        framework = self._detect_framework(language, test_files, test_dirs)

        # Get test command
        test_command = self._get_test_command(framework, language)

        # Find config files
        config_files = self._find_config_files(framework)

        # Determine if has tests
        has_tests = bool(test_files) or bool(framework)

        # Determine validation tier
        if has_tests:
            validation_tier = ValidationTier.TESTS
        else:
            validation_tier = ValidationTier.SYNTAX

        return TestDetectionResult(
            has_tests=has_tests,
            framework=framework,
            test_files=test_files,
            test_directories=test_dirs,
            test_command=test_command,
            config_files=config_files,
            validation_tier=validation_tier,
            language=language
        )

    def _detect_language(self) -> Optional[str]:
        """Detect primary programming language"""
        # Count files by extension
        extensions = {}

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not self._should_ignore_path(file_path):
                ext = file_path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1

        if not extensions:
            return None

        # Map extensions to languages
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
        }

        # Find most common language
        max_count = 0
        primary_lang = None

        for ext, count in extensions.items():
            if ext in ext_to_lang and count > max_count:
                max_count = count
                primary_lang = ext_to_lang[ext]

        return primary_lang

    def _find_test_directories(self, language: Optional[str]) -> List[str]:
        """Find test directories"""
        test_dirs = []

        # Get patterns for language
        patterns = self.TEST_DIRECTORIES.get(language, []) if language else []
        # Also check common patterns
        patterns.extend(["tests", "test", "__tests__"])
        patterns = list(set(patterns))  # Remove duplicates

        for pattern in patterns:
            dir_path = self.repo_path / pattern
            if dir_path.exists() and dir_path.is_dir():
                test_dirs.append(str(dir_path.relative_to(self.repo_path)))

        return test_dirs

    def _find_test_files(self, language: Optional[str]) -> List[str]:
        """Find test files"""
        test_files = []

        # Get patterns for language
        patterns = self.TEST_FILE_PATTERNS.get(language, []) if language else []

        for pattern in patterns:
            for file_path in self.repo_path.rglob(pattern):
                if file_path.is_file() and not self._should_ignore_path(file_path):
                    test_files.append(str(file_path.relative_to(self.repo_path)))

        return test_files[:100]  # Limit to first 100

    def _detect_framework(
        self,
        language: Optional[str],
        test_files: List[str],
        test_dirs: List[str]
    ) -> Optional[str]:
        """Detect test framework from config files or conventions"""
        # Check for framework-specific config files
        for framework, config_files in self.FRAMEWORK_CONFIGS.items():
            for config_file in config_files:
                if (self.repo_path / config_file).exists():
                    # Verify it's the right framework
                    if self._verify_framework_config(framework, config_file):
                        return framework

        # Infer from language and test files
        if language == "python":
            # Check if test files use pytest conventions
            if any("pytest" in str(f).lower() or "test_" in str(f) for f in test_files):
                return "pytest"
            return "unittest"  # Default for Python

        elif language in ("javascript", "typescript"):
            # Check package.json for test script
            package_json = self.repo_path / "package.json"
            if package_json.exists():
                try:
                    import json
                    with open(package_json) as f:
                        data = json.load(f)
                        scripts = data.get("scripts", {})
                        test_script = scripts.get("test", "")

                        if "jest" in test_script:
                            return "jest"
                        elif "mocha" in test_script:
                            return "mocha"
                        elif "vitest" in test_script:
                            return "vitest"
                except:
                    pass

            return "jest"  # Default for JS/TS

        elif language == "go":
            return "go_test"

        elif language == "rust":
            return "cargo_test"

        elif language == "ruby":
            return "rspec" if test_dirs and "spec" in test_dirs else None

        return None

    def _verify_framework_config(self, framework: str, config_file: str) -> bool:
        """Verify that config file actually configures this framework"""
        file_path = self.repo_path / config_file

        if not file_path.exists():
            return False

        # For package.json, check for framework in dependencies
        if config_file == "package.json":
            try:
                import json
                with open(file_path) as f:
                    data = json.load(f)
                    deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                    return framework in deps or framework.replace("_", "-") in deps
            except:
                return False

        # For other files, just check existence is enough
        return True

    def _get_test_command(self, framework: Optional[str], language: Optional[str]) -> Optional[str]:
        """Get command to run tests"""
        if not framework:
            return None

        commands = {
            "pytest": "pytest",
            "unittest": "python -m unittest discover",
            "jest": "npm test",
            "mocha": "npm test",
            "vitest": "npm test",
            "go_test": "go test ./...",
            "cargo_test": "cargo test",
            "rspec": "rspec",
        }

        return commands.get(framework)

    def _find_config_files(self, framework: Optional[str]) -> List[str]:
        """Find framework configuration files"""
        if not framework:
            return []

        config_files = []
        for config_file in self.FRAMEWORK_CONFIGS.get(framework, []):
            if (self.repo_path / config_file).exists():
                config_files.append(config_file)

        return config_files

    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored"""
        ignore_patterns = [
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".egg-info", "target", ".pytest_cache"
        ]

        path_str = str(path)
        return any(pattern in path_str for pattern in ignore_patterns)


class ValidationTierDetector:
    """
    Detects available validation tools (static analysis, linters, etc.)
    """

    def __init__(self, repo_path: str, language: Optional[str] = None):
        """
        Initialize detector.

        Args:
            repo_path: Path to repository
            language: Primary language (optional, will detect if not provided)
        """
        self.repo_path = Path(repo_path)
        self.language = language

    def detect_available_tiers(self) -> List[ValidationTier]:
        """
        Detect which validation tiers are available.

        Returns:
            List of available validation tiers in priority order
        """
        tiers = []

        # Static analysis
        if self._has_static_analysis():
            tiers.append(ValidationTier.STATIC_ANALYSIS)

        # Linting
        if self._has_linters():
            tiers.append(ValidationTier.LINTING)

        # Syntax checking is always available
        tiers.append(ValidationTier.SYNTAX)

        return tiers

    def _has_static_analysis(self) -> bool:
        """Check if static analysis tools are available"""
        # Check for config files
        static_configs = {
            "mypy.ini", "pyproject.toml",  # Python - mypy
            "tsconfig.json",  # TypeScript
            ".flowconfig",  # JavaScript - Flow
        }

        for config in static_configs:
            if (self.repo_path / config).exists():
                return True

        # Check if tools are installed in system
        tools = ["mypy", "pyright", "tsc", "flow"]
        for tool in tools:
            if self._is_tool_available(tool):
                return True

        return False

    def _has_linters(self) -> bool:
        """Check if linters are available"""
        # Check for config files
        linter_configs = {
            ".pylintrc", "pylintrc", ".flake8", "setup.cfg",  # Python
            ".eslintrc", ".eslintrc.json", ".eslintrc.js",  # JavaScript
            "rustfmt.toml",  # Rust
            ".rubocop.yml",  # Ruby
        }

        for config in linter_configs:
            if (self.repo_path / config).exists():
                return True

        # Check if tools are installed
        tools = ["pylint", "flake8", "eslint", "rustfmt", "rubocop"]
        for tool in tools:
            if self._is_tool_available(tool):
                return True

        return False

    def _is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available in PATH"""
        try:
            subprocess.run(
                [tool_name, "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
