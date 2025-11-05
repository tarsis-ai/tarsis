"""
Task management tool handlers.

These tools help the agent manage the task lifecycle.
"""

from typing import Dict, Any
from .base import BaseToolHandler, ToolDefinition, ToolResponse, ToolCategory


class AttemptCompletionHandler(BaseToolHandler):
    """Tool to signal task completion"""

    @property
    def name(self) -> str:
        return "attempt_completion"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TASK

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Use this tool when you believe you have successfully completed the task.

This will:
1. End the agent loop
2. Return control to the user
3. Provide a summary of what was accomplished

Only use this when:
- All code changes have been made and committed
- Pull request has been created
- You are confident the implementation is complete

If you're unsure or blocked, use ask_followup_question instead.""",
            input_schema={
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Summary of what was accomplished and any important notes"
                    },
                    "pr_url": {
                        "type": "string",
                        "description": "URL of the created pull request (if applicable)"
                    }
                },
                "required": ["result"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Signal task completion"""
        # This tool doesn't actually execute - it's a signal to end the loop
        # The agent loop checks for this tool use and breaks
        result = input_data.get("result", "Task completed")
        pr_url = input_data.get("pr_url")

        response_text = f"""✅ Task Completion

{result}"""

        if pr_url:
            response_text += f"\n\n🔗 Pull Request: {pr_url}"

        return self._success_response(response_text)


class AskFollowupQuestionHandler(BaseToolHandler):
    """Tool to ask the user for clarification"""

    @property
    def name(self) -> str:
        return "ask_followup_question"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TASK

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Ask the user a question when you need clarification or are blocked.

Use this when:
- Issue requirements are ambiguous
- You need to make a design decision
- You encounter an error you can't resolve
- You need additional context

The task will pause and wait for user input.""",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user. Be specific and provide context."
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional multiple choice options for the user"
                    }
                },
                "required": ["question"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Ask user a question"""
        question = input_data["question"]
        options = input_data.get("options", [])

        # Format the question
        formatted_question = f"❓ Question for user:\n\n{question}"

        if options:
            formatted_question += "\n\nOptions:"
            for i, option in enumerate(options, 1):
                formatted_question += f"\n{i}. {option}"

        # In a real implementation, this would pause the task and wait for user input
        # For now, we'll return a placeholder
        return self._success_response(
            formatted_question + "\n\n⏸️  Task paused. Waiting for user response..."
        )


class CreatePlanHandler(BaseToolHandler):
    """Tool to create an implementation plan"""

    @property
    def name(self) -> str:
        return "create_plan"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.TASK

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="""Create a step-by-step implementation plan for the task.

Use this early in the process to:
1. Break down the work into steps
2. Identify files that need changes
3. Plan the implementation approach
4. Get user approval before coding

The plan should be clear and detailed.""",
            input_schema={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "The implementation plan in Markdown format with numbered steps"
                    },
                    "files_to_modify": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files that will be modified"
                    },
                    "estimated_complexity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Estimated complexity of the implementation"
                    }
                },
                "required": ["plan"]
            },
            category=self.category
        )

    async def execute(self, input_data: Dict[str, Any], context: Any) -> ToolResponse:
        """Execute: Create implementation plan"""
        plan = input_data["plan"]
        files = input_data.get("files_to_modify", [])
        complexity = input_data.get("estimated_complexity", "medium")

        result = f"""📋 Implementation Plan

**Complexity:** {complexity}

{plan}
"""

        if files:
            result += "\n\n**Files to Modify:**\n"
            for file in files:
                result += f"- {file}\n"

        return self._success_response(result, metadata={"files": files, "complexity": complexity})
