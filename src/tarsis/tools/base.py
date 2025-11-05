"""
Base classes and interfaces for the tool system.

This module defines the core abstractions for tools that the agent can use
following the coordinator pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolCategory(Enum):
    """Categories of tools available to the agent"""
    GITHUB = "github"  # GitHub API operations
    GIT = "git"  # Git operations
    FILE = "file"  # File system operations
    CODE_ANALYSIS = "code_analysis"  # Code search and analysis
    TASK = "task"  # Task management (completion, clarification, etc.)


@dataclass
class ToolResponse:
    """Response from a tool execution"""
    content: Any  # The actual result content
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    def to_string(self) -> str:
        """Convert response to string format"""
        if isinstance(self.content, str):
            return self.content
        return str(self.content)


@dataclass
class ToolDefinition:
    """Definition of a tool for LLM"""
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON schema for tool inputs
    category: ToolCategory


class IToolHandler(ABC):
    """
    Base interface for all tool handlers.

    Each tool handler implements:
    1. Tool definition (name, description, schema)
    2. Execution logic
    3. Optional validation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name"""
        pass

    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """Tool category"""
        pass

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """
        Get the tool definition for the LLM.

        Returns:
            ToolDefinition with name, description, and input schema
        """
        pass

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """
        Execute the tool with given input.

        Args:
            input_data: Tool input parameters
            context: Execution context (TaskConfig, etc.)

        Returns:
            ToolResponse with results

        Raises:
            Exception if execution fails
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate tool input before execution.

        Args:
            input_data: Input to validate

        Raises:
            ValueError if validation fails
        """
        # Default implementation - can be overridden
        pass


class BaseToolHandler(IToolHandler):
    """
    Base implementation of IToolHandler with common functionality.

    Subclasses only need to implement:
    - name property
    - category property
    - get_definition()
    - execute()
    """

    def __init__(self):
        super().__init__()

    def _format_error(self, error: Exception) -> str:
        """Format an error message for returning to the LLM"""
        return f"Error executing {self.name}: {str(error)}"

    def _success_response(self, content: Any, metadata: Optional[Dict] = None) -> ToolResponse:
        """Create a successful tool response"""
        return ToolResponse(content=content, metadata=metadata)

    def _error_response(self, error: Exception) -> ToolResponse:
        """Create an error tool response"""
        return ToolResponse(
            content=self._format_error(error),
            metadata={"error": True}
        )
