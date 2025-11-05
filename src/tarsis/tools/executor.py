"""
ToolExecutor - Coordinates tool execution using the coordinator pattern.

This module manages tool registration, lookup, and execution.
"""

import logging
from typing import Dict, List, Any, Optional
from .base import IToolHandler, ToolDefinition, ToolResponse, ToolCategory

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Coordinates tool execution for the agent.

    Responsibilities:
    1. Register tool handlers
    2. Provide tool definitions to LLM
    3. Route tool calls to appropriate handlers
    4. Handle execution errors gracefully
    """

    def __init__(self):
        self._handlers: Dict[str, IToolHandler] = {}
        self._handlers_by_category: Dict[ToolCategory, List[IToolHandler]] = {
            category: [] for category in ToolCategory
        }

    def register(self, handler: IToolHandler) -> None:
        """
        Register a tool handler.

        Args:
            handler: Tool handler to register

        Raises:
            ValueError if a handler with the same name already exists
        """
        if handler.name in self._handlers:
            raise ValueError(f"Tool handler '{handler.name}' is already registered")

        self._handlers[handler.name] = handler
        self._handlers_by_category[handler.category].append(handler)
        logger.debug(f"Registered tool: {handler.name} (category: {handler.category.value})")

    def register_multiple(self, handlers: List[IToolHandler]) -> None:
        """Register multiple tool handlers at once"""
        for handler in handlers:
            self.register(handler)

    def get_tool_definitions(self, categories: Optional[List[ToolCategory]] = None) -> List[ToolDefinition]:
        """
        Get tool definitions for the LLM.

        Args:
            categories: Optional filter by categories. If None, returns all tools.

        Returns:
            List of tool definitions
        """
        if categories is None:
            # Return all tools
            return [handler.get_definition() for handler in self._handlers.values()]

        # Filter by categories
        definitions = []
        for category in categories:
            for handler in self._handlers_by_category.get(category, []):
                definitions.append(handler.get_definition())

        return definitions

    def get_tool_definitions_for_llm(self, categories: Optional[List[ToolCategory]] = None) -> List[Dict[str, Any]]:
        """
        Get tool definitions formatted for LLM API (Anthropic format).

        Returns list of tool objects with name, description, and input_schema.
        """
        definitions = self.get_tool_definitions(categories)
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in definitions
        ]

    async def execute(self, tool_use: Any, context: Any) -> ToolResponse:
        """
        Execute a tool by routing to the appropriate handler.

        Args:
            tool_use: ToolUse object with name and input
            context: Execution context (TaskConfig, etc.)

        Returns:
            ToolResponse from the handler

        Raises:
            ValueError if tool not found
            Exception if execution fails
        """
        tool_name = tool_use.name
        tool_input = tool_use.input

        logger.debug(f"Executing tool: {tool_name}")
        logger.debug(f"Tool input: {tool_input}")

        # Lookup handler
        handler = self._handlers.get(tool_name)
        if not handler:
            logger.error(f"Tool not found: {tool_name}")
            raise ValueError(
                f"Unknown tool: {tool_name}. Available tools: {list(self._handlers.keys())}"
            )

        # Validate input
        try:
            handler.validate_input(tool_input)
        except Exception as e:
            logger.error(f"Tool input validation failed for {tool_name}: {e}")
            raise ValueError(f"Invalid input for tool '{tool_name}': {str(e)}")

        # Execute tool
        try:
            result = await handler.execute(tool_input, context)

            # Log result summary
            if result.metadata and result.metadata.get("error"):
                logger.warning(f"Tool {tool_name} completed with error: {result.content}")
            else:
                # Log success with truncated content for readability
                content_str = str(result.content)
                content_preview = content_str[:200] + "..." if len(content_str) > 200 else content_str
                logger.debug(f"Tool {tool_name} completed successfully. Result preview: {content_preview}")

            return result
        except Exception as e:
            # Log detailed error
            logger.error(f"Tool {tool_name} execution failed: {e}", exc_info=True)
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            raise Exception(error_msg) from e

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered"""
        return tool_name in self._handlers

    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names"""
        return list(self._handlers.keys())

    def get_tools_by_category(self, category: ToolCategory) -> List[IToolHandler]:
        """Get all tools in a specific category"""
        return self._handlers_by_category.get(category, [])

    def clear(self) -> None:
        """Clear all registered tools (mainly for testing)"""
        self._handlers.clear()
        for category in ToolCategory:
            self._handlers_by_category[category].clear()
