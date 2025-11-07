"""
Commit grouping strategies for multi-commit support.

Provides intelligent grouping of file changes into logical commits based on:
- Commit type (feat, fix, test, docs, etc.)
- Dependencies between changes
- Size constraints

Enables splitting large changesets into semantically meaningful commits
while maintaining git history clarity.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from pathlib import Path
from collections import defaultdict

from .conventional import CommitType, detect_commit_type_from_files, detect_scope_from_files
from .message_generator import FileChange

logger = logging.getLogger(__name__)


@dataclass
class CommitGroup:
    """
    Represents a logical group of files for a single commit.

    A commit group contains files that should be committed together
    based on semantic relationships, commit type, or other grouping criteria.
    """

    files: List[FileChange]
    commit_type: CommitType
    scope: Optional[str] = None
    description_hint: Optional[str] = None  # Suggested description

    @property
    def file_count(self) -> int:
        """Number of files in this group."""
        return len(self.files)

    @property
    def total_additions(self) -> int:
        """Total lines added across all files."""
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        """Total lines deleted across all files."""
        return sum(f.deletions for f in self.files)

    @property
    def total_loc(self) -> int:
        """Total lines of code changed (additions + deletions)."""
        return self.total_additions + self.total_deletions

    @property
    def file_paths(self) -> List[str]:
        """List of all file paths in this group."""
        return [f.path for f in self.files]

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"CommitGroup({self.commit_type.value}, {self.file_count} files, {self.total_loc} LOC)"


class GroupingStrategy(ABC):
    """
    Abstract base class for commit grouping strategies.

    Strategies implement different approaches to grouping files into commits:
    - Type-based: Group by commit type (feat, fix, test, etc.)
    - Dependency-based: Order groups by logical dependencies
    - Size-based: Split large groups into smaller commits
    """

    @abstractmethod
    def group(self, files: List[FileChange]) -> List[CommitGroup]:
        """
        Group files into logical commits.

        Args:
            files: List of file changes to group

        Returns:
            List of commit groups, each representing one commit
        """
        pass

    def refine(self, groups: List[CommitGroup]) -> List[CommitGroup]:
        """
        Refine existing groups (used for multi-stage grouping).

        Default implementation returns groups unchanged.
        Subclasses can override to modify/reorder/split existing groups.

        Args:
            groups: Existing commit groups

        Returns:
            Refined commit groups
        """
        return groups


class TypeBasedGrouping(GroupingStrategy):
    """
    Group files by their natural commit type.

    Uses existing file pattern detection to determine commit type:
    - test/ → test
    - docs/ → docs
    - *.md → docs
    - src/ → feat/fix/refactor (based on change type)

    Includes smart merging:
    - Merges small groups (<min_files) with related types
    - Preserves critical separations (tests, docs, ci)
    """

    def __init__(self, min_files_per_group: int = 3, merge_threshold: int = 2):
        """
        Initialize type-based grouping strategy.

        Args:
            min_files_per_group: Minimum files to keep as separate group
            merge_threshold: Merge groups with ≤ this many files
        """
        self.min_files_per_group = min_files_per_group
        self.merge_threshold = merge_threshold

    def group(self, files: List[FileChange]) -> List[CommitGroup]:
        """Group files by commit type with smart merging."""
        if not files:
            return []

        # 1. Initial grouping by type
        type_groups = self._group_by_type(files)

        # 2. Convert to CommitGroup objects
        commit_groups = []
        for commit_type, file_list in type_groups.items():
            scope = detect_scope_from_files([f.path for f in file_list])

            group = CommitGroup(
                files=file_list,
                commit_type=commit_type,
                scope=scope,
                description_hint=self._generate_description_hint(commit_type, file_list)
            )
            commit_groups.append(group)

        # 3. Smart merging of small groups
        if len(commit_groups) > 1:
            commit_groups = self._merge_small_groups(commit_groups)

        logger.info(f"TypeBasedGrouping: Created {len(commit_groups)} groups from {len(files)} files")
        for i, group in enumerate(commit_groups):
            logger.debug(f"  Group {i+1}: {group}")

        return commit_groups

    def _group_by_type(self, files: List[FileChange]) -> Dict[CommitType, List[FileChange]]:
        """Group files by their detected commit type."""
        groups: Dict[CommitType, List[FileChange]] = defaultdict(list)

        for file_change in files:
            # Detect type using existing infrastructure
            commit_type = detect_commit_type_from_files([file_change.path])

            # Fallback: heuristic based on change type
            if not commit_type:
                if file_change.change_type == "create":
                    commit_type = CommitType.FEAT
                elif file_change.change_type == "delete":
                    commit_type = CommitType.CHORE
                elif file_change.change_type == "rename":
                    commit_type = CommitType.REFACTOR
                else:
                    commit_type = CommitType.REFACTOR

            groups[commit_type].append(file_change)

        return dict(groups)

    def _generate_description_hint(self, commit_type: CommitType, files: List[FileChange]) -> str:
        """Generate suggested description based on files."""
        if len(files) == 1:
            file_name = Path(files[0].path).name
            if files[0].change_type == "create":
                return f"add {file_name}"
            elif files[0].change_type == "delete":
                return f"remove {file_name}"
            elif files[0].change_type == "rename":
                return f"rename {files[0].old_path} to {files[0].path}"
            else:
                return f"update {file_name}"
        else:
            # Multiple files - use generic description
            scope = detect_scope_from_files([f.path for f in files])
            if scope:
                return f"update {scope} module"
            else:
                return f"update {len(files)} files"

    def _merge_small_groups(self, groups: List[CommitGroup]) -> List[CommitGroup]:
        """
        Merge small groups with related types.

        Rules:
        - Never merge: test, docs, ci (critical separations)
        - Merge style + refactor if both small
        - Merge fix + test if both small (bug fix with tests)
        - Merge small groups into largest related group
        """
        # Groups that should never be merged
        never_merge = {CommitType.TEST, CommitType.DOCS, CommitType.CI}

        # Groups that can be merged together
        mergeable_pairs = [
            {CommitType.STYLE, CommitType.REFACTOR},
            {CommitType.FIX, CommitType.TEST},
            {CommitType.FEAT, CommitType.TEST},
        ]

        # Identify small groups
        small_groups = [g for g in groups if g.file_count <= self.merge_threshold]
        large_groups = [g for g in groups if g.file_count > self.merge_threshold]

        # If only one group or all groups are large, no merging needed
        if len(groups) == 1 or not small_groups:
            return groups

        # Start with large groups
        result_groups = list(large_groups)

        # Process small groups
        for small_group in small_groups:
            # Never merge these types
            if small_group.commit_type in never_merge:
                result_groups.append(small_group)
                continue

            # Try to find a mergeable pair
            merged = False
            for pair in mergeable_pairs:
                if small_group.commit_type in pair:
                    # Find another small group in this pair
                    for other_group in result_groups:
                        if other_group.commit_type in pair and other_group.commit_type != small_group.commit_type:
                            # Merge into this group
                            other_group.files.extend(small_group.files)
                            # Update scope to cover both
                            combined_paths = [f.path for f in other_group.files]
                            other_group.scope = detect_scope_from_files(combined_paths)
                            merged = True
                            logger.debug(f"Merged {small_group.commit_type.value} into {other_group.commit_type.value}")
                            break

                if merged:
                    break

            # If not merged, keep as separate group
            if not merged:
                result_groups.append(small_group)

        return result_groups


class DependencyAwareGrouping(GroupingStrategy):
    """
    Order commit groups by logical dependencies.

    Ensures commits are made in an order that prevents breaking
    intermediate states:
    1. build/ci changes first (may affect everything else)
    2. refactor before feat/fix (API changes before consumers)
    3. feat/fix in middle (core functionality)
    4. test after feat/fix (validate changes)
    5. docs last (document completed work)
    6. style last (cosmetic)
    """

    # Priority map: lower number = earlier in sequence
    PRIORITY_MAP = {
        CommitType.BUILD: 1,
        CommitType.CI: 1,
        CommitType.REFACTOR: 2,
        CommitType.PERF: 3,
        CommitType.FEAT: 4,
        CommitType.FIX: 4,
        CommitType.TEST: 5,
        CommitType.DOCS: 6,
        CommitType.STYLE: 7,
        CommitType.CHORE: 8,
        CommitType.REVERT: 9,
    }

    def group(self, files: List[FileChange]) -> List[CommitGroup]:
        """
        This strategy doesn't create groups, only orders them.
        Use refine() instead.
        """
        raise NotImplementedError("DependencyAwareGrouping should use refine(), not group()")

    def refine(self, groups: List[CommitGroup]) -> List[CommitGroup]:
        """Order groups by dependency priority."""
        if len(groups) <= 1:
            return groups

        # Sort by priority
        sorted_groups = sorted(
            groups,
            key=lambda g: self.PRIORITY_MAP.get(g.commit_type, 99)
        )

        logger.info("DependencyAwareGrouping: Ordered groups by priority")
        for i, group in enumerate(sorted_groups):
            logger.debug(f"  {i+1}. {group.commit_type.value} ({group.file_count} files)")

        return sorted_groups


class SizeBasedGrouping(GroupingStrategy):
    """
    Split large commit groups into smaller chunks.

    Rules:
    - Groups with >max_files OR >max_loc are split
    - Preserves directory locality (files in same dir stay together)
    - Maintains semantic coherence
    """

    def __init__(self, max_files: int = 15, max_loc: int = 500):
        """
        Initialize size-based grouping strategy.

        Args:
            max_files: Maximum files per commit
            max_loc: Maximum lines of code per commit
        """
        self.max_files = max_files
        self.max_loc = max_loc

    def group(self, files: List[FileChange]) -> List[CommitGroup]:
        """
        This strategy doesn't create groups, only splits them.
        Use refine() instead.
        """
        raise NotImplementedError("SizeBasedGrouping should use refine(), not group()")

    def refine(self, groups: List[CommitGroup]) -> List[CommitGroup]:
        """Split large groups into smaller chunks."""
        result_groups = []

        for group in groups:
            if self._should_split(group):
                split_groups = self._split_group(group)
                result_groups.extend(split_groups)
                logger.info(f"SizeBasedGrouping: Split {group} into {len(split_groups)} groups")
            else:
                result_groups.append(group)

        return result_groups

    def _should_split(self, group: CommitGroup) -> bool:
        """Determine if group should be split."""
        return group.file_count > self.max_files or group.total_loc > self.max_loc

    def _split_group(self, group: CommitGroup) -> List[CommitGroup]:
        """
        Split a large group into smaller groups.

        Strategy: Group by directory to maintain locality
        """
        # Group files by directory
        by_dir: Dict[Path, List[FileChange]] = defaultdict(list)
        for file in group.files:
            dir_path = Path(file.path).parent
            by_dir[dir_path].append(file)

        # Create sub-groups respecting size limits
        sub_groups = []
        current_batch = []
        current_loc = 0

        for dir_files in by_dir.values():
            dir_loc = sum(f.additions + f.deletions for f in dir_files)

            # If adding this directory exceeds limits and we have files, flush current batch
            if current_batch and (
                len(current_batch) + len(dir_files) > self.max_files or
                current_loc + dir_loc > self.max_loc
            ):
                sub_groups.append(self._create_sub_group(current_batch, group))
                current_batch = []
                current_loc = 0

            current_batch.extend(dir_files)
            current_loc += dir_loc

        # Add remaining files
        if current_batch:
            sub_groups.append(self._create_sub_group(current_batch, group))

        return sub_groups if sub_groups else [group]

    def _create_sub_group(self, files: List[FileChange], parent_group: CommitGroup) -> CommitGroup:
        """Create a sub-group from split files."""
        scope = detect_scope_from_files([f.path for f in files])

        return CommitGroup(
            files=files,
            commit_type=parent_group.commit_type,
            scope=scope or parent_group.scope,
            description_hint=f"update {len(files)} files" if len(files) > 1 else f"update {files[0].path}"
        )


class SingleCommitGrouping(GroupingStrategy):
    """
    Fallback strategy: Put all files in a single commit.

    Used when:
    - Multi-commit adds no value (all files same type, small changeset)
    - User explicitly disables multi-commit
    - Emergency mode (quick fix needed)
    """

    def group(self, files: List[FileChange]) -> List[CommitGroup]:
        """Create a single group containing all files."""
        if not files:
            return []

        # Detect overall type and scope
        file_paths = [f.path for f in files]
        commit_type = detect_commit_type_from_files(file_paths) or CommitType.CHORE
        scope = detect_scope_from_files(file_paths)

        group = CommitGroup(
            files=files,
            commit_type=commit_type,
            scope=scope,
            description_hint=f"update {len(files)} files"
        )

        logger.info(f"SingleCommitGrouping: Created 1 group from {len(files)} files")

        return [group]


class CommitGrouper:
    """
    Orchestrates multiple grouping strategies.

    Applies strategies in sequence:
    1. Initial grouping (e.g., by type)
    2. Refinement (e.g., ordering, splitting)
    3. Validation
    """

    def __init__(
        self,
        grouping_strategy: GroupingStrategy,
        refinement_strategies: Optional[List[GroupingStrategy]] = None,
        max_groups: int = 5
    ):
        """
        Initialize commit grouper.

        Args:
            grouping_strategy: Primary strategy for initial grouping
            refinement_strategies: Optional refinement strategies (ordering, splitting)
            max_groups: Maximum number of commit groups to create
        """
        self.grouping_strategy = grouping_strategy
        self.refinement_strategies = refinement_strategies or []
        self.max_groups = max_groups

    def group_and_order(self, files: List[FileChange]) -> List[CommitGroup]:
        """
        Group files and apply refinements.

        Args:
            files: List of file changes to group

        Returns:
            Final list of commit groups, ready for commit creation
        """
        if not files:
            return []

        logger.info(f"CommitGrouper: Grouping {len(files)} files")

        # 1. Initial grouping
        groups = self.grouping_strategy.group(files)

        # 2. Apply refinement strategies
        for strategy in self.refinement_strategies:
            groups = strategy.refine(groups)

        # 3. Enforce max groups limit
        if len(groups) > self.max_groups:
            logger.warning(
                f"Generated {len(groups)} groups, exceeds max_groups={self.max_groups}. "
                f"Merging smallest groups."
            )
            groups = self._consolidate_to_limit(groups, self.max_groups)

        # 4. Validate groups
        self._validate_groups(groups, files)

        logger.info(f"CommitGrouper: Final result = {len(groups)} groups")

        return groups

    def _consolidate_to_limit(self, groups: List[CommitGroup], max_groups: int) -> List[CommitGroup]:
        """
        Merge smallest groups until we're at or below the limit.

        Strategy: Repeatedly merge the two smallest groups with compatible types
        """
        while len(groups) > max_groups:
            # Sort by size (smallest first)
            sorted_groups = sorted(groups, key=lambda g: g.file_count)

            # Merge two smallest groups
            smallest = sorted_groups[0]
            second_smallest = sorted_groups[1]

            # Create merged group
            merged_files = smallest.files + second_smallest.files
            merged_scope = detect_scope_from_files([f.path for f in merged_files])

            merged_group = CommitGroup(
                files=merged_files,
                commit_type=smallest.commit_type,  # Use first group's type
                scope=merged_scope,
                description_hint=f"update {len(merged_files)} files"
            )

            # Replace in list
            groups = [g for g in groups if g not in (smallest, second_smallest)]
            groups.append(merged_group)

            logger.debug(f"Merged {smallest.commit_type.value} + {second_smallest.commit_type.value}")

        return groups

    def _validate_groups(self, groups: List[CommitGroup], original_files: List[FileChange]) -> None:
        """
        Validate grouping results.

        Checks:
        - All files are included in exactly one group
        - No duplicate files
        - No empty groups
        """
        # Check for empty groups
        for group in groups:
            if not group.files:
                raise ValueError(f"Empty commit group found: {group}")

        # Check all files are included
        grouped_files = set()
        for group in groups:
            for file in group.files:
                if file.path in grouped_files:
                    raise ValueError(f"Duplicate file in groups: {file.path}")
                grouped_files.add(file.path)

        original_paths = {f.path for f in original_files}
        if grouped_files != original_paths:
            missing = original_paths - grouped_files
            extra = grouped_files - original_paths
            raise ValueError(
                f"Grouping validation failed. Missing: {missing}, Extra: {extra}"
            )


def should_use_multi_commit(
    files: List[FileChange],
    min_files: int = 5,
    force_single: bool = False
) -> bool:
    """
    Determine if multi-commit grouping should be used.

    Args:
        files: List of file changes
        min_files: Minimum files to trigger multi-commit
        force_single: Force single commit regardless of heuristics

    Returns:
        True if multi-commit should be used, False otherwise
    """
    if force_single:
        return False

    if len(files) < min_files:
        return False

    # Check if files are diverse enough (different types)
    file_paths = [f.path for f in files]
    types_detected = set()
    for path in file_paths:
        commit_type = detect_commit_type_from_files([path])
        if commit_type:
            types_detected.add(commit_type)

    # If all files are same type, single commit is fine
    if len(types_detected) <= 1:
        logger.info(f"All {len(files)} files are same type, single commit recommended")
        return False

    logger.info(f"{len(files)} files with {len(types_detected)} different types, multi-commit recommended")
    return True
