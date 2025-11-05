"""
AgentTask - Core agent loop implementation using a recursive pattern.

This module implements the main agentic loop that:
1. Maintains conversation history
2. Makes recursive LLM requests
3. Executes tools based on LLM responses
4. Manages task state and context
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class TaskConfig:
    """Configuration for a task execution"""
    issue_number: int
    repo_owner: str
    repo_name: str
    default_branch: str
    max_iterations: int = 25
    max_consecutive_mistakes: int = 3
    auto_approve: bool = False


@dataclass
class Message:
    """Represents a message in the conversation history"""
    role: str  # "user", "assistant", "system"
    content: Any  # Can be string or list of content blocks


@dataclass
class ToolUse:
    """Represents a tool use request from the LLM"""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution"""
    tool_use_id: str
    content: Any
    is_error: bool = False


class AgentTask:
    """
    Core agent task that implements the recursive LLM loop pattern.

    This class manages:
    - Conversation history with the LLM
    - Tool execution coordination
    - Task state and progress tracking
    - Iteration limits and safety checks
    """

    def __init__(
        self,
        config: TaskConfig,
        llm_provider: Any,  # Will be typed properly when we create the provider interface
        tool_executor: Any,  # Will be typed properly when we create the tool executor
        state_manager: Any = None
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.tool_executor = tool_executor
        self.state_manager = state_manager

        # Task state
        self.status = TaskStatus.PENDING
        self.conversation_history: List[Message] = []
        self.iteration_count = 0
        self.consecutive_mistakes = 0
        self.abort_requested = False

        # Context tracking
        self.files_accessed: set = set()
        self.files_modified: set = set()
        self.branch_name: Optional[str] = None
        self.pr_url: Optional[str] = None

        # Validation tracking
        self.validation_performed: bool = False
        self.validation_passed: bool = False
        self.last_validation_iteration: Optional[int] = None

    async def execute(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Main entry point to execute the task.

        Args:
            initial_prompt: The initial user prompt (e.g., issue description)

        Returns:
            Dict containing task results (PR URL, files changed, etc.)
        """
        logger.info(f"Starting task execution for issue #{self.config.issue_number}")
        logger.debug(f"Initial prompt: {initial_prompt[:200]}...")

        self.status = TaskStatus.IN_PROGRESS

        try:
            # Build initial user message
            user_content = self._build_initial_message(initial_prompt)

            # Start the recursive loop
            await self._initiate_task_loop(user_content)

            # Task completed successfully
            self.status = TaskStatus.COMPLETED
            logger.info(f"Task completed successfully in {self.iteration_count} iterations")
            return self._build_task_result()

        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.error(f"Task failed after {self.iteration_count} iterations: {e}", exc_info=True)
            raise

    async def _initiate_task_loop(self, user_content: Any) -> None:
        """
        Main task loop - recursively makes LLM requests and executes tools.

        This implements the core agentic pattern:
        1. Send message to LLM
        2. Parse response for tool uses
        3. Execute tools
        4. Send tool results back to LLM
        5. Repeat until completion or abort
        """
        next_user_content = user_content
        include_file_details = True

        while not self.abort_requested and self.iteration_count < self.config.max_iterations:
            # Check if we've made too many consecutive mistakes
            if self.consecutive_mistakes >= self.config.max_consecutive_mistakes:
                raise Exception(
                    f"Stopping: {self.consecutive_mistakes} consecutive mistakes. "
                    "Please review and provide guidance."
                )

            # Make LLM request and execute tools
            did_end_loop = await self._recursively_make_requests(
                next_user_content,
                include_file_details
            )

            # After first iteration, don't re-include file details
            include_file_details = False

            if did_end_loop:
                # Task completed via attempt_completion tool
                break
            else:
                # Model didn't use any tools, prompt it to continue
                next_user_content = [{
                    "type": "text",
                    "text": "Please continue with the next step or use the attempt_completion tool if you're done."
                }]

        # Check if we hit iteration limit
        if self.iteration_count >= self.config.max_iterations:
            raise Exception(f"Task aborted: Reached maximum iteration limit ({self.config.max_iterations})")

    async def _recursively_make_requests(
        self,
        user_content: Any,
        include_file_details: bool = False
    ) -> bool:
        """
        Make a single LLM request, execute tools, and handle responses.

        Args:
            user_content: User message content to send
            include_file_details: Whether to include accessed file details in context

        Returns:
            True if task loop should end (via attempt_completion), False otherwise
        """
        self.iteration_count += 1

        logger.info(f"=== Iteration {self.iteration_count}/{self.config.max_iterations} ===")

        # Add user message to history
        self.conversation_history.append(Message(
            role="user",
            content=user_content
        ))

        # Build system prompt
        system_prompt = await self._build_system_prompt(include_file_details)
        logger.debug(f"System prompt length: {len(system_prompt)} chars")

        # Get tool definitions
        tools = self.tool_executor.get_tool_definitions_for_llm()
        logger.debug(f"Available tools: {len(tools)}")

        # Make LLM request
        logger.debug("Sending request to LLM...")
        assistant_message = await self.llm_provider.create_message(
            system_prompt=system_prompt,
            messages=self._format_messages_for_llm(),
            tools=tools
        )

        logger.debug(f"Received LLM response: stop_reason={assistant_message.stop_reason}")

        # Add assistant response to history
        self.conversation_history.append(Message(
            role="assistant",
            content=assistant_message.content
        ))

        # Parse and execute tools
        tool_uses = self._extract_tool_uses(assistant_message.content)

        # Debug logging
        logger.info(f"Iteration {self.iteration_count}: Extracted {len(tool_uses)} tool uses")
        if tool_uses:
            for tool_use in tool_uses:
                logger.info(f"  - Tool: {tool_use.name}")
        else:
            logger.warning(f"No tools extracted from response. Content type: {type(assistant_message.content)}")
            logger.warning(f"{assistant_message.content}")
            if isinstance(assistant_message.content, list):
                for i, block in enumerate(assistant_message.content):
                    if isinstance(block, dict):
                        logger.debug(f"  Content block {i}: type={block.get('type')}, keys={list(block.keys())}")
                        if block.get('type') == 'text':
                            logger.debug(f"    Text preview: {block.get('text', '')[:200]}")

        if not tool_uses:
            # No tools used - will prompt model to continue
            return False

        # Execute tools and collect results
        tool_results = []
        did_use_attempt_completion = False

        for tool_use in tool_uses:
            # Check for completion tool
            if tool_use.name == "attempt_completion":
                did_use_attempt_completion = True
                # Store completion message
                self.completion_message = tool_use.input.get("result", "Task completed")
                logger.info(f"Task completion requested: {self.completion_message}")
                continue

            # Execute tool
            try:
                logger.debug(f"Executing tool: {tool_use.name}")
                # Pass self (AgentTask) as context so tools can access validation state
                result = await self.tool_executor.execute(tool_use, self)
                tool_results.append(ToolResult(
                    tool_use_id=tool_use.id,
                    content=result.content,
                    is_error=False
                ))

                # Track successful tool use
                self.consecutive_mistakes = 0
                logger.debug(f"Tool {tool_use.name} executed successfully")

                # Track validation state
                if tool_use.name == "run_validation":
                    self.validation_performed = True
                    self.last_validation_iteration = self.iteration_count
                    # Check if validation passed (look for "passed", "success", or "skipped" in result)
                    result_str = str(result.content).lower()
                    self.validation_passed = (
                        "passed" in result_str or
                        "success" in result_str or
                        "skipped" in result_str  # Treat skipped as pass (e.g., no local clone yet)
                    )
                    logger.info(f"Validation performed: passed={self.validation_passed}")

                # Reset validation state if files are modified after validation
                if tool_use.name in ["modify_file", "commit_changes"]:
                    if self.validation_performed:
                        logger.info("Files modified after validation - validation state reset")
                        self.validation_performed = False
                        self.validation_passed = False

            except Exception as e:
                # Tool execution failed
                logger.warning(f"Tool {tool_use.name} failed: {e}")
                error_message = f"Tool execution failed: {str(e)}"
                tool_results.append(ToolResult(
                    tool_use_id=tool_use.id,
                    content=error_message,
                    is_error=True
                ))

                # Track mistake
                self.consecutive_mistakes += 1
                logger.warning(f"Consecutive mistakes: {self.consecutive_mistakes}/{self.config.max_consecutive_mistakes}")

        # If we have tool results, add them to conversation
        if tool_results and not did_use_attempt_completion:
            self.conversation_history.append(Message(
                role="user",
                content=self._format_tool_results(tool_results)
            ))

        return did_use_attempt_completion

    def _build_initial_message(self, prompt: str) -> List[Dict[str, str]]:
        """Build the initial user message content"""
        return [{
            "type": "text",
            "text": prompt
        }]

    async def _build_system_prompt(self, include_file_details: bool = False) -> str:
        """
        Build the system prompt for the LLM using the modular prompt builder.
        """
        from ..prompts import PromptBuilder

        builder = PromptBuilder()

        # Add context information
        builder.add_context_section(
            "TASK_CONTEXT",
            f"""## Current Task

**Repository**: {self.config.repo_owner}/{self.config.repo_name}
**Issue**: #{self.config.issue_number}
**Branch**: {self.branch_name or 'Not created yet'}
**Iteration**: {self.iteration_count}/{self.config.max_iterations}
"""
        )

        # Add file context if requested
        if include_file_details and self.files_accessed:
            file_list = "\n".join(f"- {f}" for f in self.files_accessed)
            builder.add_context_section(
                "FILE_CONTEXT",
                f"""## Files Accessed
{file_list}
"""
            )

        # Build the prompt
        return builder.build()

    def _format_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Format conversation history for LLM API"""
        formatted = []
        for msg in self.conversation_history:
            formatted.append({
                "role": msg.role,
                "content": msg.content
            })
        return formatted

    def _extract_tool_uses(self, content: Any) -> List[ToolUse]:
        """
        Extract tool use requests from assistant message content.

        Content can be a string or a list of content blocks.
        Tool uses are represented as blocks with type="tool_use".
        """
        tool_uses = []

        if isinstance(content, str):
            # Plain text response, no tools
            return tool_uses

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_uses.append(ToolUse(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {})
                    ))

        return tool_uses

    def _format_tool_results(self, results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Format tool results as user message content"""
        formatted = []
        for result in results:
            formatted.append({
                "type": "tool_result",
                "tool_use_id": result.tool_use_id,
                "content": result.content,
                "is_error": result.is_error
            })
        return formatted

    def _build_task_result(self) -> Dict[str, Any]:
        """Build the final task result"""
        return {
            "status": self.status.value,
            "iterations": self.iteration_count,
            "files_modified": list(self.files_modified),
            "branch_name": self.branch_name,
            "pr_url": self.pr_url,
            "completion_message": getattr(self, "completion_message", "")
        }

    def abort(self) -> None:
        """Request task abort"""
        self.abort_requested = True
        self.status = TaskStatus.ABORTED
