"""
Reflexion Framework Implementation for Tarsis

This module implements the Reflexion framework for language agents,
enabling self-reflection and learning from mistakes.

Based on the paper: "Reflexion: Language Agents with Verbal Reinforcement Learning"
(arXiv:2303.11366v4)

Key components:
- ReflectionEntry: Single reflection event
- ReflectionMemory: FIFO episodic memory buffer
- ReflectionManager: Orchestrates reflection process
- Integration with AgentTask for autonomous learning
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from collections import Counter

logger = logging.getLogger(__name__)


class ReflectionTrigger(Enum):
    """Types of events that trigger self-reflection"""

    VALIDATION_FAILURE = "validation_failure"
    TOOL_ERROR = "tool_error"
    CONSECUTIVE_MISTAKES = "consecutive_mistakes"
    PERIODIC = "periodic"
    TRIAL_FAILURE = "trial_failure"  # For multi-trial mode
    PRE_COMPLETION = "pre_completion"  # Before attempting task completion


@dataclass
class ReflectionEntry:
    """
    A single self-reflection event.

    Represents one instance where the agent reflected on its performance
    and generated insights for improvement.

    Attributes:
        iteration: Agent iteration when reflection occurred
        trigger: What triggered this reflection
        context: Context data (error details, validation results, etc.)
        insight: LLM-generated reflection text (the "learning")
        timestamp: When this reflection was created
        applied: Whether this insight has been used in decision-making
    """

    iteration: int
    trigger: ReflectionTrigger
    context: Dict[str, Any]
    insight: str
    timestamp: str
    applied: bool = False

    def __post_init__(self):
        """Ensure trigger is ReflectionTrigger enum"""
        if isinstance(self.trigger, str):
            self.trigger = ReflectionTrigger(self.trigger)


@dataclass
class ReflectionConfig:
    """Configuration for Reflexion framework"""

    # Core settings
    enabled: bool = True
    mode: str = "within_task"  # within_task, multi_trial, or hybrid
    memory_size: int = 10
    temperature: float = 0.5

    # Trigger settings
    trigger_validation_failure: bool = True
    trigger_tool_error: bool = True
    trigger_consecutive_mistakes: int = 3  # 0 = disabled
    trigger_periodic_interval: int = 5  # 0 = disabled
    trigger_pre_completion: bool = True  # Verify requirements before completion

    # Multi-trial settings
    max_trials: int = 5

    # Repository-level learning
    persist_across_issues: bool = False
    repo_cache_dir: str = ".tarsis/reflections"

    @classmethod
    def from_env(cls, env_dict: Dict[str, str]) -> "ReflectionConfig":
        """Create config from environment variables"""

        def get_bool(key: str, default: bool) -> bool:
            val = env_dict.get(key, str(default)).lower()
            return val in ("true", "1", "yes")

        def get_int(key: str, default: int) -> int:
            try:
                return int(env_dict.get(key, str(default)))
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            try:
                return float(env_dict.get(key, str(default)))
            except ValueError:
                return default

        return cls(
            enabled=get_bool("REFLEXION_ENABLED", True),
            mode=env_dict.get("REFLEXION_MODE", "within_task"),
            memory_size=get_int("REFLEXION_MEMORY_SIZE", 10),
            temperature=get_float("REFLEXION_TEMPERATURE", 0.5),
            trigger_validation_failure=get_bool("REFLEXION_TRIGGER_VALIDATION_FAILURE", True),
            trigger_tool_error=get_bool("REFLEXION_TRIGGER_TOOL_ERROR", True),
            trigger_consecutive_mistakes=get_int("REFLEXION_TRIGGER_CONSECUTIVE_MISTAKES", 3),
            trigger_periodic_interval=get_int("REFLEXION_TRIGGER_PERIODIC_INTERVAL", 5),
            trigger_pre_completion=get_bool("REFLEXION_TRIGGER_PRE_COMPLETION", True),
            max_trials=get_int("REFLEXION_MAX_TRIALS", 5),
            persist_across_issues=get_bool("REFLEXION_PERSIST_ACROSS_ISSUES", False),
            repo_cache_dir=env_dict.get("REFLEXION_REPO_CACHE_DIR", ".tarsis/reflections"),
        )


class ReflectionMemory:
    """
    Episodic memory buffer for reflections.

    Maintains a FIFO (First-In-First-Out) buffer of reflection entries
    with configurable maximum size. When capacity is exceeded, oldest
    reflections are automatically removed.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize reflection memory.

        Args:
            max_size: Maximum number of reflections to store
        """
        self.max_size = max_size
        self.entries: List[ReflectionEntry] = []

    def add(self, entry: ReflectionEntry) -> None:
        """
        Add reflection to memory with sliding window.

        Args:
            entry: Reflection entry to add
        """
        self.entries.append(entry)

        # Enforce max size (FIFO)
        if len(self.entries) > self.max_size:
            removed = self.entries.pop(0)
            logger.debug(f"Memory full, removed oldest reflection from iteration {removed.iteration}")

    def has_reflections(self) -> bool:
        """Check if memory contains any reflections"""
        return len(self.entries) > 0

    def get_recent(self, limit: int = 3) -> List[ReflectionEntry]:
        """
        Get most recent reflections.

        Args:
            limit: Maximum number to return

        Returns:
            List of recent reflection entries
        """
        return self.entries[-limit:]

    def get_by_trigger(self, trigger: ReflectionTrigger) -> List[ReflectionEntry]:
        """
        Get reflections by trigger type.

        Args:
            trigger: Trigger type to filter by

        Returns:
            List of matching reflection entries
        """
        return [e for e in self.entries if e.trigger == trigger]

    def format_for_prompt(self) -> str:
        """
        Format reflections for system prompt injection.

        Returns:
            Formatted string suitable for LLM context
        """
        if not self.entries:
            return "No previous reflections in this task."

        formatted = "## LESSONS LEARNED FROM PREVIOUS ATTEMPTS\n\n"

        # Group by trigger type
        by_trigger = self._group_by_trigger()

        for trigger, entries in by_trigger.items():
            formatted += f"### {trigger.value.replace('_', ' ').title()}\n\n"

            # Show last 3 per trigger type
            for entry in entries[-3:]:
                formatted += f"**Iteration {entry.iteration}**:\n"
                formatted += f"{entry.insight}\n\n"

        return formatted

    def format_for_context(self, limit: int = 3) -> str:
        """
        Format for inclusion in reflection prompts.

        Args:
            limit: Maximum number of reflections to include

        Returns:
            Formatted string for reflection prompt context
        """
        if not self.entries:
            return "None (first attempt)"

        formatted = ""
        for entry in self.entries[-limit:]:
            formatted += f"[Iteration {entry.iteration} - {entry.trigger.value}]\n"
            formatted += f"{entry.insight}\n\n"

        return formatted

    def _group_by_trigger(self) -> Dict[ReflectionTrigger, List[ReflectionEntry]]:
        """Group reflections by trigger type"""
        groups: Dict[ReflectionTrigger, List[ReflectionEntry]] = {}
        for entry in self.entries:
            if entry.trigger not in groups:
                groups[entry.trigger] = []
            groups[entry.trigger].append(entry)
        return groups

    def clear(self) -> None:
        """Clear all reflections from memory"""
        self.entries = []
        logger.debug("Reflection memory cleared")

    def seed_from_cache(self, cached_reflections: List[ReflectionEntry], limit: int = 3) -> None:
        """
        Seed memory with cached reflections from repository.

        Args:
            cached_reflections: Reflections loaded from cache
            limit: Maximum number to seed
        """
        # Prioritize validation failures and recent reflections
        sorted_reflections = sorted(
            cached_reflections,
            key=lambda r: (
                r.trigger == ReflectionTrigger.VALIDATION_FAILURE,
                r.timestamp
            ),
            reverse=True
        )

        for reflection in sorted_reflections[:limit]:
            # Mark as not applied (from cache)
            reflection.applied = False
            self.add(reflection)

        logger.info(f"Seeded memory with {len(self.entries)} cached reflections")


class ReflectionParser:
    """Parse and extract structured insights from reflections"""

    @staticmethod
    def extract_action_items(reflection_text: str) -> List[str]:
        """
        Extract concrete action items from reflection.

        Args:
            reflection_text: LLM-generated reflection

        Returns:
            List of extracted action items
        """
        action_items = []

        # Look for numbered lists, bullet points, "should", "need to", etc.
        patterns = [
            r'\d+\.\s*\*\*[^:]+\*\*:\s*(.+)',  # Numbered with bold headers
            r'[-*]\s*(.+)',  # Bullet points
            r'(?:should|must|need to)\s+(.+?)(?:\.|$)',  # Action verbs
        ]

        for pattern in patterns:
            matches = re.findall(pattern, reflection_text, re.IGNORECASE | re.MULTILINE)
            action_items.extend([m.strip() for m in matches if len(m.strip()) > 10])

        # Deduplicate and return top 5
        seen = set()
        unique_items = []
        for item in action_items:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique_items.append(item)

        return unique_items[:5]

    @staticmethod
    def extract_patterns(reflection_text: str) -> List[str]:
        """
        Extract identified patterns from reflection.

        Args:
            reflection_text: LLM-generated reflection

        Returns:
            List of extracted patterns
        """
        # Look for "pattern", "repeatedly", "similar mistake", etc.
        pattern_indicators = [
            r'pattern[^.!?]*[.!?]',
            r'repeatedly[^.!?]*[.!?]',
            r'similar[^.!?]*mistake[^.!?]*[.!?]',
            r'keep[^.!?]*(?:making|doing)[^.!?]*[.!?]',
        ]

        patterns = []
        for indicator in pattern_indicators:
            matches = re.findall(indicator, reflection_text, re.IGNORECASE)
            patterns.extend([m.strip() for m in matches])

        return patterns[:3]  # Top 3 patterns

    @staticmethod
    def extract_key_lesson(insight: str) -> str:
        """
        Extract one-line summary from reflection.

        Args:
            insight: Full reflection text

        Returns:
            One-line summary
        """
        # Look for first substantive sentence or bullet point
        lines = insight.split('\n')
        for line in lines:
            stripped = line.strip()
            # Skip headers and short lines
            if stripped and len(stripped) > 30 and not stripped.endswith(':'):
                # Remove markdown formatting
                cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', stripped)
                cleaned = re.sub(r'\* ', '', cleaned)
                return cleaned[:150]  # Truncate to 150 chars

        # Fallback: return first 150 chars
        return insight[:150].strip()


class ReflectionManager:
    """
    Orchestrates the Reflexion process.

    Manages reflection triggering, LLM calls for self-reflection,
    memory storage, and integration with the agent's decision-making.
    """

    def __init__(self, llm_provider, config: ReflectionConfig):
        """
        Initialize reflection manager.

        Args:
            llm_provider: LLM provider instance for reflection calls
            config: Reflexion configuration
        """
        self.llm_provider = llm_provider
        self.config = config
        self.memory = ReflectionMemory(max_size=config.memory_size)

        # Repository cache (initialized later if needed)
        self.repo_cache = None

        # Metrics
        self.reflection_count = 0
        self.triggers_by_type: Counter = Counter()

    async def initialize(self, repo_owner: str, repo_name: str):
        """
        Initialize reflection manager for a task.

        Loads past reflections from repository cache if enabled.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
        """
        logger.info("🧠 Initializing Reflexion framework")

        # Initialize repository cache if persistence enabled
        if self.config.persist_across_issues:
            try:
                from ..repository.reflection_cache import ReflectionCache
                self.repo_cache = ReflectionCache(self.config.repo_cache_dir)

                # Load past reflections
                logger.info("📚 Loading repository-level reflections...")
                past_reflections = self.repo_cache.load_reflections(
                    repo_owner, repo_name, max_age_days=30
                )

                if past_reflections:
                    self.memory.seed_from_cache(past_reflections, limit=3)
                    logger.info(f"✅ Loaded {len(self.memory.entries)} past reflections")
                else:
                    logger.info("No past reflections found in cache")

            except ImportError:
                logger.warning("ReflectionCache not yet implemented, skipping repository learning")
            except Exception as e:
                logger.error(f"Failed to load reflection cache: {e}")

    async def finalize(self, repo_owner: str, repo_name: str, issue_number: str):
        """
        Finalize reflection manager after task completion.

        Saves reflections to repository cache if enabled.

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            issue_number: Issue number
        """
        if not self.repo_cache or not self.config.persist_across_issues:
            return

        if not self.memory.has_reflections():
            logger.info("No reflections to save")
            return

        try:
            logger.info("💾 Saving reflections to repository cache...")
            self.repo_cache.save_reflections(
                repo_owner, repo_name, issue_number, self.memory.entries
            )
            logger.info("✅ Reflections saved for future issues")
        except Exception as e:
            logger.error(f"Failed to save reflections to cache: {e}")

    async def trigger_reflection(
        self,
        trigger: ReflectionTrigger,
        context: Dict[str, Any],
        conversation_history: List[Any]
    ) -> Optional[ReflectionEntry]:
        """
        Trigger self-reflection and store insight.

        Args:
            trigger: What triggered this reflection
            context: Context data (error details, validation results, etc.)
            conversation_history: Agent's conversation history

        Returns:
            Generated reflection entry, or None if reflection failed
        """
        logger.info(f"🧠 Reflection triggered: {trigger.value}")
        logger.debug(f"Context: {context}")

        try:
            # Build reflection prompt
            prompt = self._build_reflection_prompt(trigger, context, conversation_history)

            # Generate reflection via LLM
            reflection_text = await self._generate_reflection(prompt)

            # Create entry
            entry = ReflectionEntry(
                iteration=context.get("iteration", 0),
                trigger=trigger,
                context=context,
                insight=reflection_text,
                timestamp=datetime.now().isoformat(),
                applied=False
            )

            # Store in memory
            self.memory.add(entry)

            # Update metrics
            self.reflection_count += 1
            self.triggers_by_type[trigger.value] += 1

            # Log insight summary
            insight_preview = reflection_text[:150] + "..." if len(reflection_text) > 150 else reflection_text
            logger.info(f"💡 Insight: {insight_preview}")
            logger.info(f"📊 Memory: {len(self.memory.entries)}/{self.config.memory_size}")

            return entry

        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}", exc_info=True)
            return None

    async def _generate_reflection(self, prompt: str) -> str:
        """
        Make LLM call for reflection.

        Args:
            prompt: Reflection prompt

        Returns:
            LLM-generated reflection text
        """
        # Local import
        from ..prompts.reflection_prompts import REFLECTION_SYSTEM_PROMPT

        response = await self.llm_provider.create_message(
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            tools=None,  # No tool calling during reflection
            temperature=self.config.temperature,  # 0.5 for creativity
            max_tokens=2048  # Longer reflections
        )

        # Extract content (handle both string and structured responses)
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            # Multiple content blocks, join text blocks
            text_blocks = [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in response.content
            ]
            return "\n".join(text_blocks)
        else:
            return str(response.content)

    def _build_reflection_prompt(
        self,
        trigger: ReflectionTrigger,
        context: Dict[str, Any],
        conversation_history: List[Any]
    ) -> str:
        """
        Build context-specific reflection prompt.

        Args:
            trigger: Trigger type
            context: Context data
            conversation_history: Agent's conversation history

        Returns:
            Formatted reflection prompt
        """
        # Local import
        from ..prompts.reflection_prompts import (
            VALIDATION_FAILURE_PROMPT,
            TOOL_ERROR_PROMPT,
            CONSECUTIVE_MISTAKES_PROMPT,
            PERIODIC_CHECKPOINT_PROMPT,
            TRIAL_FAILURE_PROMPT,
            PRE_COMPLETION_PROMPT
        )

        # Get appropriate template
        template_map = {
            ReflectionTrigger.VALIDATION_FAILURE: VALIDATION_FAILURE_PROMPT,
            ReflectionTrigger.TOOL_ERROR: TOOL_ERROR_PROMPT,
            ReflectionTrigger.CONSECUTIVE_MISTAKES: CONSECUTIVE_MISTAKES_PROMPT,
            ReflectionTrigger.PERIODIC: PERIODIC_CHECKPOINT_PROMPT,
            ReflectionTrigger.TRIAL_FAILURE: TRIAL_FAILURE_PROMPT,
            ReflectionTrigger.PRE_COMPLETION: PRE_COMPLETION_PROMPT,
        }

        template = template_map.get(trigger, PERIODIC_CHECKPOINT_PROMPT)

        # Extract recent actions from conversation history
        recent_actions = self._extract_recent_actions(conversation_history, limit=5)

        # Format previous reflections
        previous_reflections = self.memory.format_for_context()

        # Prepare context with defaults
        prompt_context = {
            "recent_actions": recent_actions,
            "previous_reflections": previous_reflections,
            **context  # Merge in provided context
        }

        # Handle missing keys gracefully
        try:
            return template.format(**prompt_context)
        except KeyError as e:
            logger.warning(f"Missing key in reflection prompt: {e}, using defaults")
            # Fill in missing keys with empty strings
            for key in re.findall(r'\{(\w+)\}', template):
                if key not in prompt_context:
                    prompt_context[key] = "N/A"
            return template.format(**prompt_context)

    def _extract_recent_actions(self, conversation_history: List[Any], limit: int = 5) -> str:
        """
        Extract recent actions from conversation history.

        Args:
            conversation_history: Agent's conversation history
            limit: Number of recent messages to extract

        Returns:
            Formatted string of recent actions
        """
        if not conversation_history:
            return "No recent actions available."

        recent_actions = []

        # Get last N messages
        for msg in conversation_history[-limit:]:
            # Handle tool uses
            if hasattr(msg, 'tool_uses') and msg.tool_uses:
                for tool_use in msg.tool_uses:
                    action = f"- Used tool: {tool_use.name}"
                    if hasattr(tool_use, 'input') and tool_use.input:
                        # Truncate input for readability
                        input_summary = str(tool_use.input)[:100]
                        action += f" (input: {input_summary}...)"
                    recent_actions.append(action)

            # Handle user messages
            elif hasattr(msg, 'role') and msg.role == 'user':
                content = str(msg.content)[:150]
                recent_actions.append(f"- User message: {content}...")

        if not recent_actions:
            return "No tool actions found in recent history."

        return "\n".join(recent_actions)

    def has_reflections(self) -> bool:
        """Check if manager has any reflections"""
        return self.memory.has_reflections()

    def count_applied_lessons(self) -> int:
        """Count how many reflections were marked as applied"""
        return sum(1 for entry in self.memory.entries if entry.applied)

    def get_learning_summary(self) -> str:
        """
        Generate human-readable learning summary for PR.

        Returns:
            Markdown-formatted summary
        """
        if not self.memory.has_reflections():
            return "No reflections generated (task completed on first attempt)."

        summary = "The AI agent reflected on its performance and learned:\n\n"

        # Group by trigger
        by_trigger = self.memory._group_by_trigger()

        for trigger, entries in by_trigger.items():
            summary += f"**{trigger.value.replace('_', ' ').title()}** ({len(entries)} times):\n"

            # Extract key insights
            for entry in entries[:2]:  # Top 2 per trigger
                key_lesson = ReflectionParser.extract_key_lesson(entry.insight)
                summary += f"- {key_lesson}\n"

            summary += "\n"

        return summary
