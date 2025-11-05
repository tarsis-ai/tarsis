"""
File type detection and categorization utilities.

Provides functionality to identify file types, programming languages,
and categorize files for better repository analysis.
"""

from enum import Enum
from typing import Optional, Set
from pathlib import Path


class FileCategory(str, Enum):
    """File category classifications"""
    SOURCE_CODE = "source_code"
    TEST = "test"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    BUILD = "build"
    DATA = "data"
    ASSET = "asset"
    DEPENDENCY = "dependency"
    SCRIPT = "script"
    UNKNOWN = "unknown"


class Language(str, Enum):
    """Programming language classifications"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SHELL = "shell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    UNKNOWN = "unknown"


class FileTypeDetector:
    """
    Detects file types, categories, and programming languages.

    Uses extension-based detection and path pattern matching.
    """

    # Extension to language mapping
    LANGUAGE_EXTENSIONS = {
        ".py": Language.PYTHON,
        ".pyw": Language.PYTHON,
        ".pyi": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".java": Language.JAVA,
        ".c": Language.C,
        ".h": Language.C,
        ".cpp": Language.CPP,
        ".cc": Language.CPP,
        ".cxx": Language.CPP,
        ".hpp": Language.CPP,
        ".cs": Language.CSHARP,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".kts": Language.KOTLIN,
        ".sh": Language.SHELL,
        ".bash": Language.SHELL,
        ".zsh": Language.SHELL,
        ".sql": Language.SQL,
        ".html": Language.HTML,
        ".htm": Language.HTML,
        ".css": Language.CSS,
        ".scss": Language.CSS,
        ".sass": Language.CSS,
        ".less": Language.CSS,
        ".md": Language.MARKDOWN,
        ".markdown": Language.MARKDOWN,
        ".json": Language.JSON,
        ".yaml": Language.YAML,
        ".yml": Language.YAML,
        ".xml": Language.XML,
    }

    # Extension to category mapping
    CATEGORY_EXTENSIONS = {
        # Configuration
        ".json": FileCategory.CONFIGURATION,
        ".yaml": FileCategory.CONFIGURATION,
        ".yml": FileCategory.CONFIGURATION,
        ".toml": FileCategory.CONFIGURATION,
        ".ini": FileCategory.CONFIGURATION,
        ".conf": FileCategory.CONFIGURATION,
        ".config": FileCategory.CONFIGURATION,
        ".env": FileCategory.CONFIGURATION,

        # Documentation
        ".md": FileCategory.DOCUMENTATION,
        ".markdown": FileCategory.DOCUMENTATION,
        ".rst": FileCategory.DOCUMENTATION,
        ".txt": FileCategory.DOCUMENTATION,
        ".adoc": FileCategory.DOCUMENTATION,

        # Build/Deploy
        ".dockerfile": FileCategory.BUILD,
        ".dockerignore": FileCategory.BUILD,
        ".gitignore": FileCategory.BUILD,
        ".yml": FileCategory.BUILD,  # Could be CI/CD

        # Assets
        ".png": FileCategory.ASSET,
        ".jpg": FileCategory.ASSET,
        ".jpeg": FileCategory.ASSET,
        ".gif": FileCategory.ASSET,
        ".svg": FileCategory.ASSET,
        ".ico": FileCategory.ASSET,

        # Data
        ".csv": FileCategory.DATA,
        ".tsv": FileCategory.DATA,
        ".parquet": FileCategory.DATA,
        ".db": FileCategory.DATA,
        ".sqlite": FileCategory.DATA,
    }

    # File name patterns for special files
    CONFIG_FILES = {
        "package.json", "package-lock.json", "yarn.lock",
        "requirements.txt", "pipfile", "pipfile.lock", "pyproject.toml", "setup.py",
        "cargo.toml", "cargo.lock",
        "go.mod", "go.sum",
        "gemfile", "gemfile.lock",
        "composer.json", "composer.lock",
        ".gitignore", ".dockerignore", ".npmignore",
        "tsconfig.json", "jsconfig.json",
        "webpack.config.js", "vite.config.js",
        "dockerfile", "docker-compose.yml",
        ".env", ".env.example", ".env.local",
        "makefile", "rakefile",
        ".editorconfig", ".prettierrc", ".eslintrc",
    }

    BUILD_FILES = {
        "dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "makefile", "rakefile", "build.gradle", "pom.xml",
        ".gitlab-ci.yml", ".travis.yml", "jenkinsfile",
        ".github/workflows",
    }

    DOC_FILES = {
        "readme.md", "readme.txt", "readme",
        "license", "license.md", "license.txt",
        "contributing.md", "changelog.md", "history.md",
        "authors", "contributors",
    }

    # Test directory patterns
    TEST_PATTERNS = {"test", "tests", "__tests__", "spec", "specs"}

    # Binary extensions (non-text files)
    BINARY_EXTENSIONS = {
        ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
        ".exe", ".bin", ".dat",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
        ".mp3", ".mp4", ".avi", ".mov", ".wav",
        ".woff", ".woff2", ".ttf", ".eot",
    }

    @classmethod
    def detect_language(cls, file_path: str) -> Language:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected language
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        return cls.LANGUAGE_EXTENSIONS.get(ext, Language.UNKNOWN)

    @classmethod
    def detect_category(cls, file_path: str) -> FileCategory:
        """
        Detect file category based on extension and path patterns.

        Args:
            file_path: Path to the file

        Returns:
            Detected file category
        """
        path = Path(file_path)
        name_lower = path.name.lower()
        ext = path.suffix.lower()

        # Check if it's in a test directory
        parts = [p.lower() for p in path.parts]
        if any(test_pattern in parts for test_pattern in cls.TEST_PATTERNS):
            if ext in cls.LANGUAGE_EXTENSIONS:  # Source code in test dir
                return FileCategory.TEST

        # Check special filenames
        if name_lower in cls.CONFIG_FILES:
            return FileCategory.CONFIGURATION

        if name_lower in cls.BUILD_FILES:
            return FileCategory.BUILD

        if name_lower in cls.DOC_FILES:
            return FileCategory.DOCUMENTATION

        # Check by extension
        if ext in cls.CATEGORY_EXTENSIONS:
            return cls.CATEGORY_EXTENSIONS[ext]

        # Check if it's source code
        if ext in cls.LANGUAGE_EXTENSIONS:
            lang = cls.LANGUAGE_EXTENSIONS[ext]
            # Script languages
            if lang in {Language.SHELL, Language.PYTHON, Language.RUBY}:
                if any(part in {"scripts", "bin", "tools"} for part in parts):
                    return FileCategory.SCRIPT
            return FileCategory.SOURCE_CODE

        # Binary files
        if ext in cls.BINARY_EXTENSIONS:
            if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"}:
                return FileCategory.ASSET
            return FileCategory.DATA

        return FileCategory.UNKNOWN

    @classmethod
    def is_binary(cls, file_path: str) -> bool:
        """
        Check if file is likely binary (non-text).

        Args:
            file_path: Path to the file

        Returns:
            True if file is likely binary
        """
        ext = Path(file_path).suffix.lower()
        return ext in cls.BINARY_EXTENSIONS

    @classmethod
    def is_source_code(cls, file_path: str) -> bool:
        """
        Check if file is source code.

        Args:
            file_path: Path to the file

        Returns:
            True if file is source code
        """
        category = cls.detect_category(file_path)
        return category in {FileCategory.SOURCE_CODE, FileCategory.TEST, FileCategory.SCRIPT}

    @classmethod
    def get_source_extensions(cls, language: Optional[Language] = None) -> Set[str]:
        """
        Get file extensions for source code files.

        Args:
            language: Optional language filter

        Returns:
            Set of file extensions
        """
        if language is None:
            return set(cls.LANGUAGE_EXTENSIONS.keys())

        return {ext for ext, lang in cls.LANGUAGE_EXTENSIONS.items() if lang == language}

    @classmethod
    def should_exclude(cls, file_path: str) -> bool:
        """
        Check if file should be excluded from analysis.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be excluded
        """
        path = Path(file_path)
        parts = [p.lower() for p in path.parts]

        # Exclude patterns
        exclude_dirs = {
            "node_modules", ".git", ".svn", ".hg",
            "__pycache__", ".pytest_cache", ".mypy_cache",
            "venv", "env", ".venv", ".env",
            "dist", "build", "target",
            ".idea", ".vscode", ".vs",
            "coverage", ".coverage",
        }

        # Check if any part of the path matches exclude patterns
        if any(part in exclude_dirs for part in parts):
            return True

        # Exclude compiled/generated files
        if cls.is_binary(file_path) and not path.suffix in {".png", ".jpg", ".svg"}:
            return True

        return False
