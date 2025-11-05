"""
Handler for repositories without tests.

Provides user interaction when no tests are found, allowing the user to decide how to proceed.
"""

from enum import Enum
from typing import Optional, List
from dataclasses import dataclass

from .detector import TestDetectionResult, ValidationTier


class NoTestsDecision(Enum):
    """User decision when no tests are found"""
    PROCEED_WITH_VALIDATION = "proceed"  # Proceed with fallback validation
    CREATE_TESTS = "create"  # Ask agent to create tests
    SKIP_VALIDATION = "skip"  # Skip validation entirely
    ABORT = "abort"  # Abort the task


@dataclass
class NoTestsConfig:
    """Configuration for no-tests behavior"""
    default_behavior: str = "ask"  # ask, proceed, skip, abort
    allow_user_override: bool = True
    suggest_test_creation: bool = True
    available_fallback_tiers: List[ValidationTier] = None


class NoTestsHandler:
    """
    Handles the scenario when a repository has no tests.

    Provides user interaction via ask_followup_question tool to determine how to proceed.
    """

    def __init__(self, config: Optional[NoTestsConfig] = None):
        """
        Initialize handler.

        Args:
            config: Configuration for behavior (defaults to asking user)
        """
        self.config = config or NoTestsConfig()

    def should_ask_user(self, detection_result: TestDetectionResult) -> bool:
        """
        Determine if we should ask the user for input.

        Args:
            detection_result: Result of test detection

        Returns:
            True if user input is needed
        """
        # If tests exist, no need to ask
        if detection_result.has_tests:
            return False

        # If explicit behavior specified (not "ask"), don't ask - just use it
        # This prevents infinite loops when agent sets no_tests_behavior='proceed'
        if self.config.default_behavior != "ask":
            return False

        # Only ask if default_behavior is "ask"
        return True

    def get_default_decision(self, detection_result: TestDetectionResult) -> NoTestsDecision:
        """
        Get the default decision based on configuration.

        Args:
            detection_result: Result of test detection

        Returns:
            Default decision
        """
        behavior_map = {
            "proceed": NoTestsDecision.PROCEED_WITH_VALIDATION,
            "skip": NoTestsDecision.SKIP_VALIDATION,
            "abort": NoTestsDecision.ABORT,
            "ask": NoTestsDecision.PROCEED_WITH_VALIDATION  # Default to proceed if not asking
        }

        return behavior_map.get(
            self.config.default_behavior,
            NoTestsDecision.PROCEED_WITH_VALIDATION
        )

    def build_question_text(self, detection_result: TestDetectionResult) -> str:
        """
        Build the question text to present to the user.

        Args:
            detection_result: Result of test detection

        Returns:
            Formatted question text
        """
        language = detection_result.language or "this project"

        question = f"""⚠️ No tests found in {language} repository.

**Detection Results:**
- Language: {detection_result.language or 'Unknown'}
- Test directories searched: {', '.join(detection_result.test_directories) if detection_result.test_directories else 'None found'}
- Test files found: {len(detection_result.test_files)}
"""

        # Add available fallback tiers
        if detection_result.available_tiers:
            tiers_text = ", ".join([tier.value for tier in detection_result.available_tiers])
            question += f"\n**Available fallback validation:**\n- {tiers_text}\n"

        question += "\n**How would you like to proceed?**"

        return question

    def get_question_options(self, detection_result: TestDetectionResult) -> List[str]:
        """
        Get multiple choice options for the user.

        Args:
            detection_result: Result of test detection

        Returns:
            List of option strings
        """
        options = []

        # Option 1: Proceed with fallback validation
        if detection_result.available_tiers:
            fallback_tier = detection_result.available_tiers[0].value if detection_result.available_tiers else "syntax checking"
            options.append(f"Proceed with fallback validation ({fallback_tier})")
        else:
            options.append("Proceed with syntax checking only")

        # Option 2: Create tests (if enabled in config)
        if self.config.suggest_test_creation:
            options.append("Ask the agent to create basic tests first")

        # Option 3: Skip validation
        options.append("Skip validation and create PR anyway")

        # Option 4: Abort
        options.append("Abort this task")

        return options

    def parse_user_response(self, user_response: str, options: List[str]) -> NoTestsDecision:
        """
        Parse user response into a decision.

        Args:
            user_response: User's response text
            options: The options that were presented

        Returns:
            NoTestsDecision based on response
        """
        response_lower = user_response.lower().strip()

        # Map responses to decisions
        if "proceed" in response_lower or "fallback" in response_lower or "validation" in response_lower:
            return NoTestsDecision.PROCEED_WITH_VALIDATION
        elif "create" in response_lower or "test" in response_lower and "create" in options[1].lower() if len(options) > 1 else False:
            return NoTestsDecision.CREATE_TESTS
        elif "skip" in response_lower:
            return NoTestsDecision.SKIP_VALIDATION
        elif "abort" in response_lower or "cancel" in response_lower:
            return NoTestsDecision.ABORT

        # Default to proceed if unclear
        return NoTestsDecision.PROCEED_WITH_VALIDATION

    def format_decision_explanation(self, decision: NoTestsDecision, detection_result: TestDetectionResult) -> str:
        """
        Format an explanation of the decision for logging/reporting.

        Args:
            decision: The decision that was made
            detection_result: Detection results

        Returns:
            Explanation text
        """
        explanations = {
            NoTestsDecision.PROCEED_WITH_VALIDATION: (
                f"Proceeding with fallback validation. "
                f"Available tiers: {', '.join([t.value for t in detection_result.available_tiers]) if detection_result.available_tiers else 'syntax only'}"
            ),
            NoTestsDecision.CREATE_TESTS: (
                "Agent will attempt to create basic tests before proceeding with implementation."
            ),
            NoTestsDecision.SKIP_VALIDATION: (
                "Skipping validation. Changes will be committed without automated validation."
            ),
            NoTestsDecision.ABORT: (
                "Task aborted due to lack of tests."
            )
        }

        return explanations.get(decision, "Unknown decision")

    def create_question_for_tool(self, detection_result: TestDetectionResult) -> dict:
        """
        Create the question payload for the ask_followup_question tool.

        Args:
            detection_result: Result of test detection

        Returns:
            Dictionary with 'question' and 'options' keys for the tool
        """
        return {
            "question": self.build_question_text(detection_result),
            "options": self.get_question_options(detection_result)
        }
