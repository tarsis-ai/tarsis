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
import os
import uuid
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
        llm_provider: Any,
        tool_executor: Any,
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
        self.consecutive_empty_responses = 0  # Track empty responses to detect stuck state
        self.abort_requested = False

        # Context tracking
        self.files_accessed: set = set()
        self.files_modified: set = set()
        self.branch_name: Optional[str] = None
        self.pr_url: Optional[str] = None
        self.tools_used_count: Dict[str, int] = {}  # Track tool usage counts

        # Validation tracking
        self.validation_performed: bool = False
        self.validation_passed: bool = False
        self.last_validation_iteration: Optional[int] = None

        # Local repository clone management
        self.clone_manager: Optional[Any] = None
        self._initialize_clone_manager()

        # Reflexion framework state
        self.reflection_manager: Optional[Any] = None
        self.reflection_config: Optional[Any] = None  # Store config for access
        self.reflection_mode: str = "within_task"  # within_task, multi_trial, or hybrid
        self.trial_number: int = 0  # For multi-trial mode
        self.last_reflection_iteration: Optional[int] = None
        self.completion_message: Optional[str] = None  # For trial success detection
        self.original_task_description: Optional[str] = None  # Store for pre-completion verification
        self._initialize_reflection_manager()

    def _initialize_clone_manager(self) -> None:
        """Initialize the clone manager for local repository operations."""
        try:
            from ..repository import CloneManager

            # Get GitHub token from environment
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                logger.warning("GITHUB_TOKEN not found - clone manager disabled")
                return

            # Create clone manager with unique task ID
            task_id = str(uuid.uuid4())
            self.clone_manager = CloneManager(
                owner=self.config.repo_owner,
                name=self.config.repo_name,
                token=github_token,
                task_id=task_id
            )
            logger.info(f"Clone manager initialized for {self.config.repo_owner}/{self.config.repo_name}")
        except ImportError as e:
            logger.warning(f"CloneManager not available: {e}")
            logger.info("Install GitPython to enable local repository operations: pip install GitPython")
        except Exception as e:
            logger.warning(f"Failed to initialize clone manager: {e}")

    def _initialize_reflection_manager(self) -> None:
        """Initialize the Reflexion framework for self-reflection and learning."""
        try:
            from .reflection import ReflectionManager, ReflectionConfig

            # Load configuration from environment
            config = ReflectionConfig.from_env(os.environ)

            # Store config for access in tool handlers
            self.reflection_config = config

            # Skip initialization if disabled
            if not config.enabled:
                logger.info("Reflexion framework disabled via configuration")
                return

            # Create reflection manager
            self.reflection_manager = ReflectionManager(
                llm_provider=self.llm_provider,
                config=config
            )
            self.reflection_mode = config.mode

            logger.info(f"🧠 Reflexion framework initialized (mode: {config.mode})")
        except ImportError as e:
            logger.warning(f"ReflectionManager not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize reflection manager: {e}")

    async def execute(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Main entry point to execute the task.

        Supports multiple Reflexion modes:
        - within_task: Reflect during single task execution (default)
        - multi_trial: Trial-based learning with full resets
        - hybrid: Within-task first, escalate to multi-trial if needed

        Args:
            initial_prompt: The initial user prompt (e.g., issue description)

        Returns:
            Dict containing task results (PR URL, files changed, etc.)
        """
        logger.info(f"Starting task execution for issue #{self.config.issue_number}")
        logger.debug(f"Initial prompt: {initial_prompt[:200]}...")

        # Store original task description for pre-completion verification
        self.original_task_description = initial_prompt

        try:
            # REFLEXION: Initialize reflection manager
            if self.reflection_manager:
                await self.reflection_manager.initialize(
                    self.config.repo_owner,
                    self.config.repo_name
                )

            # Determine execution mode
            mode = self.reflection_mode if self.reflection_manager else "standard"

            if mode == "multi_trial":
                logger.info("🧠 Executing in multi-trial Reflexion mode")
                return await self.execute_with_trials(initial_prompt)

            elif mode == "hybrid":
                logger.info("🧠 Executing in hybrid Reflexion mode (within-task → multi-trial)")
                # Try within-task first
                result = await self._execute_within_task(initial_prompt)

                # If failed, escalate to multi-trial
                if not self._is_trial_successful():
                    logger.info("🔄 Hybrid mode: Escalating to multi-trial after failure")
                    self._reset_for_next_trial()
                    self.trial_number = 1  # First within-task counts as trial 1
                    return await self.execute_with_trials(initial_prompt)

                return result

            else:  # within_task or standard
                if mode == "within_task":
                    logger.info("🧠 Executing in within-task Reflexion mode")
                else:
                    logger.info("Executing in standard mode (Reflexion disabled)")

                return await self._execute_within_task(initial_prompt)

        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.error(f"Task failed after {self.iteration_count} iterations: {e}", exc_info=True)
            raise

        finally:
            # REFLEXION: Finalize reflection manager (save to cache if enabled)
            if self.reflection_manager:
                try:
                    issue_number = str(self.config.issue_number)
                    await self.reflection_manager.finalize(
                        self.config.repo_owner,
                        self.config.repo_name,
                        issue_number
                    )
                except Exception as e:
                    logger.warning(f"Failed to finalize reflection manager: {e}")

            # Cleanup local clone
            if self.clone_manager:
                try:
                    logger.info("Cleaning up local repository clone...")
                    await self.clone_manager.cleanup()
                except Exception as e:
                    logger.warning(f"Failed to cleanup clone: {e}")

    async def _execute_within_task(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Execute task in within-task mode (standard execution with reflection).

        Args:
            initial_prompt: Task description

        Returns:
            Task result dict
        """
        self.status = TaskStatus.IN_PROGRESS

        # Build initial user message
        user_content = self._build_initial_message(initial_prompt)

        # Start the recursive loop
        await self._initiate_task_loop(user_content)

        # Task completed successfully
        self.status = TaskStatus.COMPLETED
        logger.info(f"Task completed successfully in {self.iteration_count} iterations")
        return self._build_task_result()

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
                # REFLEXION: Trigger reflection before aborting due to consecutive mistakes
                if self.reflection_manager:
                    from .reflection import ReflectionTrigger
                    await self.reflection_manager.trigger_reflection(
                        trigger=ReflectionTrigger.CONSECUTIVE_MISTAKES,
                        context={
                            "mistake_count": self.consecutive_mistakes,
                            "recent_errors": self._get_recent_errors(),
                            "iteration": self.iteration_count,
                            "pattern": "repeated_failures"
                        },
                        conversation_history=self.conversation_history
                    )
                    self.last_reflection_iteration = self.iteration_count

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

        # REFLEXION: Periodic checkpoint reflection
        if (self.reflection_manager and
            self.iteration_count % 5 == 0 and  # Every 5 iterations (configurable)
            self.iteration_count != self.last_reflection_iteration):

            from .reflection import ReflectionTrigger
            await self.reflection_manager.trigger_reflection(
                trigger=ReflectionTrigger.PERIODIC,
                context={
                    "iteration": self.iteration_count,
                    "files_accessed": len(self.files_accessed),
                    "files_modified": len(self.files_modified),
                    "validation_performed": self.validation_performed,
                    "validation_passed": self.validation_passed,
                    "tools_used": self._format_tools_used()
                },
                conversation_history=self.conversation_history
            )
            self.last_reflection_iteration = self.iteration_count

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
            # No tools used - track consecutive empty responses
            self.consecutive_empty_responses += 1
            logger.warning(f"Consecutive empty responses: {self.consecutive_empty_responses}")

            # If stuck (too many consecutive empty responses), abort
            MAX_EMPTY_RESPONSES = 5
            if self.consecutive_empty_responses >= MAX_EMPTY_RESPONSES:
                logger.error(f"Model appears stuck - {MAX_EMPTY_RESPONSES} consecutive empty responses. Aborting.")
                raise Exception(
                    f"Task aborted: Model returned {MAX_EMPTY_RESPONSES} consecutive empty responses. "
                    "This usually indicates the model is stuck or confused. "
                    "Try using a different LLM provider or model."
                )

            # Prompt model to continue
            return False

        # Reset consecutive empty responses counter when tools are used
        self.consecutive_empty_responses = 0

        # Execute tools and collect results
        tool_results = []
        did_use_attempt_completion = False

        for tool_use in tool_uses:
            # Check for completion tool
            if tool_use.name == "attempt_completion":
                # REFLEXION: Trigger pre-completion verification
                if self.reflection_manager and self.reflection_config.trigger_pre_completion:
                    from .reflection import ReflectionTrigger
                    logger.info("🧠 Triggering pre-completion verification reflection")

                    await self.reflection_manager.trigger_reflection(
                        trigger=ReflectionTrigger.PRE_COMPLETION,
                        context={
                            "original_task": self.original_task_description,
                            "iterations_used": self.iteration_count,
                            "files_modified": list(self.files_modified),
                            "validation_performed": self.validation_performed,
                            "validation_passed": self.validation_passed,
                            "tools_used": dict(self.tools_used_count),
                            "modified_files_list": "\n".join([f"- {f}" for f in self.files_modified]) if self.files_modified else "None",
                            "completion_message": tool_use.input.get("result", "Task completed")
                        },
                        conversation_history=self.conversation_history
                    )
                    self.last_reflection_iteration = self.iteration_count

                    # Get the latest reflection to check if task is truly complete
                    if self.reflection_manager.memory.entries:
                        latest_reflection = self.reflection_manager.memory.entries[-1]
                        reflection_text = latest_reflection.insight.lower()

                        # Check if reflection indicates incompleteness
                        # Look for keywords that suggest missing requirements
                        incomplete_indicators = [
                            "incomplete", "missing", "not created", "haven't",
                            "did not", "didn't", "should have", "need to",
                            "required but", "not all", "partially"
                        ]

                        is_incomplete = any(indicator in reflection_text for indicator in incomplete_indicators)

                        if is_incomplete:
                            logger.warning("⚠️ Pre-completion verification detected incomplete requirements")
                            # Add reflection to conversation as a reminder
                            self.conversation_history.append(Message(
                                role="user",
                                content=[{
                                    "type": "text",
                                    "text": f"""⚠️ **Task Not Yet Complete**

Your pre-completion verification revealed that the task is INCOMPLETE:

{latest_reflection.insight}

**You must address these missing requirements before calling attempt_completion again.**

Please continue working on the task."""
                                }]
                            ))
                            # Don't mark as completed - continue the loop
                            continue
                        else:
                            logger.info("✅ Pre-completion verification passed - all requirements met")

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

                # Track tool usage count for reflections
                self.tools_used_count[tool_use.name] = self.tools_used_count.get(tool_use.name, 0) + 1

                # Track context from tool metadata
                self._update_context_from_tool_result(tool_use.name, result)

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

                    # REFLEXION: Trigger reflection on validation failure
                    if not self.validation_passed and self.reflection_manager:
                        from .reflection import ReflectionTrigger
                        await self.reflection_manager.trigger_reflection(
                            trigger=ReflectionTrigger.VALIDATION_FAILURE,
                            context={
                                "validation_result": result.metadata if hasattr(result, 'metadata') else {},
                                "iteration": self.iteration_count,
                                "files_modified": list(self.files_modified),
                                "validation_summary": str(result.content)[:500],
                                "failed_tests": str(result.content)[:1000],
                                "lint_issues": "See validation summary",
                                "static_errors": "See validation summary"
                            },
                            conversation_history=self.conversation_history
                        )
                        self.last_reflection_iteration = self.iteration_count

                # Reset validation state if files are modified after validation
                if tool_use.name in ["modify_file", "commit_changes", "modify_files_local"]:
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

                # REFLEXION: Trigger reflection on tool error
                if self.reflection_manager:
                    from .reflection import ReflectionTrigger
                    await self.reflection_manager.trigger_reflection(
                        trigger=ReflectionTrigger.TOOL_ERROR,
                        context={
                            "tool_name": tool_use.name,
                            "error_message": str(e),
                            "error_type": type(e).__name__,
                            "tool_input": str(tool_use.input)[:500],
                            "iteration": self.iteration_count,
                            "consecutive_mistakes": self.consecutive_mistakes
                        },
                        conversation_history=self.conversation_history
                    )
                    self.last_reflection_iteration = self.iteration_count

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

        # REFLEXION: Add reflection insights if available
        if self.reflection_manager and self.reflection_manager.has_reflections():
            reflection_context = self.reflection_manager.memory.format_for_prompt()
            builder.add_context_section("LEARNING_FROM_EXPERIENCE", reflection_context)

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

    def _update_context_from_tool_result(self, tool_name: str, result: Any) -> None:
        """
        Update agent context tracking based on tool execution results.

        Args:
            tool_name: Name of the tool that was executed
            result: ToolResponse from the tool execution
        """
        # Extract metadata if available
        metadata = getattr(result, 'metadata', None)
        if not metadata:
            return

        # Track branch creation
        if tool_name == "create_branch":
            if "branch_name" in metadata:
                old_branch = self.branch_name
                self.branch_name = metadata["branch_name"]
                logger.info(f"Context updated: branch_name = '{self.branch_name}' (was: '{old_branch}')")

        # Track file modifications from modify_file
        if tool_name == "modify_file":
            if "file_path" in metadata:
                file_path = metadata["file_path"]
                self.files_modified.add(file_path)
                logger.info(f"Context updated: added '{file_path}' to files_modified (total: {len(self.files_modified)})")

        # Track file modifications from commit_changes
        if tool_name == "commit_changes":
            # Check for 'files' or 'files_modified' in metadata
            files = metadata.get("files") or metadata.get("files_modified")
            if files and isinstance(files, list):
                for file_path in files:
                    self.files_modified.add(file_path)
                logger.info(f"Context updated: added {len(files)} files to files_modified (total: {len(self.files_modified)})")

        # Track file modifications from modify_files_local (batch operations)
        if tool_name == "modify_files_local":
            if "files_modified" in metadata:
                files = metadata["files_modified"]
                if isinstance(files, list):
                    for file_path in files:
                        self.files_modified.add(file_path)
                    logger.info(f"Context updated: added {len(files)} files to files_modified (total: {len(self.files_modified)})")

        # Track PR creation
        if tool_name == "create_pull_request":
            if "pr_url" in metadata:
                self.pr_url = metadata["pr_url"]
                logger.info(f"Context updated: pr_url = '{self.pr_url}'")

        # Track file accesses
        if tool_name == "read_file":
            if "file_path" in metadata:
                file_path = metadata["file_path"]
                self.files_accessed.add(file_path)
                logger.debug(f"Context updated: added '{file_path}' to files_accessed")

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

    def _get_recent_errors(self) -> List[str]:
        """Extract recent errors from conversation history for reflection"""
        errors = []
        # Look at last 10 messages for error patterns
        for msg in self.conversation_history[-10:]:
            content_str = str(msg.content).lower()
            if any(keyword in content_str for keyword in ["error", "failed", "exception", "traceback"]):
                # Truncate long error messages
                error_snippet = str(msg.content)[:200]
                errors.append(error_snippet)

        return errors if errors else ["No specific error messages captured"]

    def _format_tools_used(self) -> str:
        """Format tool usage counts for reflection context"""
        tool_counts = {}

        # Count tool uses from conversation history
        for msg in self.conversation_history:
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        tool_name = block.get('name', 'unknown')
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        if not tool_counts:
            return "No tools used yet"

        # Format as list
        formatted = []
        for tool_name, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"{tool_name}: {count} times")

        return "\n".join(formatted)

    def _is_trial_successful(self) -> bool:
        """
        Determine if current trial succeeded.

        Returns:
            True if trial completed successfully
        """
        success_criteria = [
            self.status == TaskStatus.COMPLETED or self.completion_message is not None,
            self.validation_passed or not self.validation_performed,
            not self.abort_requested
        ]

        return all(success_criteria)

    def _reset_for_next_trial(self) -> None:
        """Reset state between trials while preserving learning"""
        logger.info(f"🔄 Resetting for trial {self.trial_number + 1}")

        # KEEP (preserve across trials):
        # - self.reflection_manager (learning persists!)
        # - self.trial_number (incremented in execute_with_trials)
        # - self.config
        # - self.llm_provider
        # - self.tool_executor
        # - self.clone_manager

        # RESET (fresh state for new trial):
        self.conversation_history = []
        self.iteration_count = 0
        self.consecutive_mistakes = 0
        self.files_accessed = set()
        self.files_modified = set()
        self.branch_name = None
        self.pr_url = None
        self.validation_performed = False
        self.validation_passed = False
        self.last_validation_iteration = None
        self.last_reflection_iteration = None
        self.completion_message = None
        self.abort_requested = False
        self.status = TaskStatus.PENDING

        logger.debug("State reset complete, reflections preserved")

    async def _reflect_on_trial_failure(self, trial_num: int) -> None:
        """Generate comprehensive reflection on failed trial"""
        logger.info(f"🧠 Reflecting on trial {trial_num + 1} failure...")

        if not self.reflection_manager:
            return

        from .reflection import ReflectionTrigger

        # Build comprehensive trial summary
        trial_summary = {
            "trial_number": trial_num + 1,
            "iterations_used": self.iteration_count,
            "files_modified": list(self.files_modified),
            "validation_performed": self.validation_performed,
            "validation_passed": self.validation_passed,
            "abort_reason": "consecutive_mistakes" if self.abort_requested else "incomplete",
            "completion_attempted": self.completion_message is not None,
            "tools_used": self._format_tools_used(),
            "key_decisions": "See conversation history",
            "full_conversation": f"Trial used {self.iteration_count} iterations"
        }

        # Trigger comprehensive trial reflection
        await self.reflection_manager.trigger_reflection(
            trigger=ReflectionTrigger.TRIAL_FAILURE,
            context=trial_summary,
            conversation_history=self.conversation_history
        )

    async def execute_with_trials(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Execute task with multi-trial Reflexion approach.

        The agent will attempt the task multiple times, learning from
        each failure through self-reflection.

        Args:
            initial_prompt: Initial task description

        Returns:
            Task result dict
        """
        max_trials = 5  # Default, will be configurable
        if self.reflection_manager:
            max_trials = self.reflection_manager.config.max_trials

        for trial in range(max_trials):
            self.trial_number = trial + 1
            logger.info(f"🔄 Trial {self.trial_number}/{max_trials} starting")

            # Add trial context to prompt for subsequent trials
            trial_prompt = initial_prompt
            if trial > 0:
                trial_context = f"\n\n**TRIAL {self.trial_number}**: You have attempted this task {trial} time(s) before and it did not succeed. Review your reflections carefully and try a different approach."
                trial_prompt = initial_prompt + trial_context

            # Run single trial
            try:
                result = await self._run_single_trial(trial_prompt)

                # Evaluate success
                if self._is_trial_successful():
                    logger.info(f"✅ Success on trial {self.trial_number}!")
                    return {
                        **result,
                        "trials_used": self.trial_number,
                        "learning_applied": True,
                        "reflexion_mode": "multi_trial"
                    }
            except Exception as e:
                logger.warning(f"Trial {self.trial_number} failed with exception: {e}")
                # Continue to next trial

            # Don't reflect on last trial (no point)
            if trial < max_trials - 1:
                # Generate comprehensive trial-level reflection
                await self._reflect_on_trial_failure(trial)

                # Reset state for next trial
                self._reset_for_next_trial()

        logger.warning(f"⚠️ Max trials ({max_trials}) reached without success")
        return {
            "status": "failed",
            "trials_used": max_trials,
            "success": False,
            "reason": "max_trials_exceeded",
            "reflexion_mode": "multi_trial"
        }

    async def _run_single_trial(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Run a single trial (standard agent execution).

        Args:
            initial_prompt: Task description (may include trial context)

        Returns:
            Task result dict
        """
        # Reset iteration-specific state
        self.status = TaskStatus.IN_PROGRESS

        # Build initial user message
        user_content = self._build_initial_message(initial_prompt)

        # Execute standard agent loop
        await self._initiate_task_loop(user_content)

        # Build result
        return self._build_task_result()

    def abort(self) -> None:
        """Request task abort"""
        self.abort_requested = True
        self.status = TaskStatus.ABORTED
