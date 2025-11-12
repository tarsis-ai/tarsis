"""
Reflection Metrics - Track Reflexion framework performance

Provides metrics for monitoring the effectiveness of self-reflection and learning:
- Trigger frequencies
- Trial success rates
- Memory utilization
- Cross-issue learning rates

These metrics help evaluate whether Reflexion is improving agent performance.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ReflectionMetrics:
    """
    Track Reflexion framework performance metrics.

    These metrics help answer questions like:
    - How often does the agent reflect?
    - Does multi-trial mode improve success?
    - Is repository-level learning effective?
    - What triggers reflection most frequently?
    """

    # Trigger statistics
    triggers_by_type: Counter = field(default_factory=Counter)

    # Trial statistics (multi-trial mode)
    trials_to_success: List[int] = field(default_factory=list)
    total_trials: int = 0
    successful_trials: int = 0

    # Memory statistics
    max_memory_size: int = 0
    avg_memory_size: float = 0.0
    memory_size_samples: List[int] = field(default_factory=list)

    # Learning statistics
    reflections_generated: int = 0
    reflections_applied: int = 0
    cross_issue_hits: int = 0  # Repository cache hits

    def record_trigger(self, trigger_type: str) -> None:
        """
        Record a reflection trigger.

        Args:
            trigger_type: Type of trigger (validation_failure, tool_error, etc.)
        """
        self.triggers_by_type[trigger_type] += 1
        logger.debug(f"Recorded trigger: {trigger_type}")

    def record_trial_result(self, trial_number: int, success: bool) -> None:
        """
        Record multi-trial result.

        Args:
            trial_number: Which trial succeeded (or failed)
            success: Whether the trial succeeded
        """
        self.total_trials += 1

        if success:
            self.successful_trials += 1
            self.trials_to_success.append(trial_number)
            logger.info(f"Trial {trial_number} succeeded (metric recorded)")

    def record_memory_size(self, size: int) -> None:
        """
        Record current memory size.

        Args:
            size: Number of reflections in memory
        """
        self.memory_size_samples.append(size)
        self.max_memory_size = max(self.max_memory_size, size)

        if self.memory_size_samples:
            self.avg_memory_size = sum(self.memory_size_samples) / len(self.memory_size_samples)

    def record_reflection(self, applied: bool = False) -> None:
        """
        Record reflection generation.

        Args:
            applied: Whether this reflection was applied in decision-making
        """
        self.reflections_generated += 1
        if applied:
            self.reflections_applied += 1

    def record_cache_hit(self) -> None:
        """Record successful cross-issue learning (repository cache hit)"""
        self.cross_issue_hits += 1
        logger.debug("Repository cache hit recorded")

    def get_summary(self) -> Dict[str, any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dict with all metrics
        """
        return {
            # Trigger statistics
            "triggers": dict(self.triggers_by_type),
            "total_triggers": sum(self.triggers_by_type.values()),

            # Trial statistics
            "avg_trials_to_success": (
                sum(self.trials_to_success) / len(self.trials_to_success)
                if self.trials_to_success else 0
            ),
            "success_rate": (
                self.successful_trials / self.total_trials
                if self.total_trials > 0 else 0
            ),
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,

            # Memory statistics
            "avg_memory_size": self.avg_memory_size,
            "max_memory_size": self.max_memory_size,

            # Learning statistics
            "reflections_generated": self.reflections_generated,
            "reflections_applied": self.reflections_applied,
            "cross_issue_learning": self.cross_issue_hits,

            # Application rate
            "application_rate": (
                self.reflections_applied / self.reflections_generated
                if self.reflections_generated > 0 else 0
            )
        }

    def get_formatted_summary(self) -> str:
        """
        Get human-readable metrics summary.

        Returns:
            Formatted string for logging or display
        """
        summary = self.get_summary()

        report = "📊 Reflexion Metrics Summary\n"
        report += "=" * 50 + "\n\n"

        report += "Triggers:\n"
        for trigger, count in summary["triggers"].items():
            report += f"  - {trigger}: {count}\n"
        report += f"  Total: {summary['total_triggers']}\n\n"

        report += "Multi-Trial Performance:\n"
        report += f"  - Avg trials to success: {summary['avg_trials_to_success']:.2f}\n"
        report += f"  - Success rate: {summary['success_rate']:.1%}\n"
        report += f"  - Total trials: {summary['total_trials']}\n\n"

        report += "Memory Usage:\n"
        report += f"  - Avg memory size: {summary['avg_memory_size']:.1f}\n"
        report += f"  - Max memory size: {summary['max_memory_size']}\n\n"

        report += "Learning:\n"
        report += f"  - Reflections generated: {summary['reflections_generated']}\n"
        report += f"  - Reflections applied: {summary['reflections_applied']}\n"
        report += f"  - Cross-issue learning hits: {summary['cross_issue_learning']}\n"
        report += f"  - Application rate: {summary['application_rate']:.1%}\n"

        return report

    def reset(self) -> None:
        """Reset all metrics (for new task)"""
        self.triggers_by_type = Counter()
        self.trials_to_success = []
        self.total_trials = 0
        self.successful_trials = 0
        self.max_memory_size = 0
        self.avg_memory_size = 0.0
        self.memory_size_samples = []
        self.reflections_generated = 0
        self.reflections_applied = 0
        self.cross_issue_hits = 0
        logger.debug("Metrics reset")


# Global metrics instance (can be accessed across modules)
_global_metrics = ReflectionMetrics()


def get_metrics() -> ReflectionMetrics:
    """Get global metrics instance"""
    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics"""
    _global_metrics.reset()
