"""
Tools module - Provides all tool handlers for the agent.
"""

from .base import (
    IToolHandler,
    BaseToolHandler,
    ToolDefinition,
    ToolResponse,
    ToolCategory
)
from .executor import ToolExecutor
from .github_tools import (
    ReadIssueHandler,
    CreateBranchHandler,
    ModifyFileHandler,
    CommitChangesHandler,
    CreatePullRequestHandler,
    PostCommentHandler
)
from .file_tools import (
    ReadFileHandler,
    ListFilesHandler,
    SearchFilesHandler,
    GetRepositoryOverviewHandler
)
from .search_tools import (
    SearchCodeHandler,
    FindSymbolHandler,
    GrepPatternHandler
)
from .discovery_tools import (
    DiscoverRelevantFilesHandler
)
from .task_tools import (
    AttemptCompletionHandler,
    AskFollowupQuestionHandler,
    CreatePlanHandler
)
from .validation_tools import (
    RunValidationHandler
)
from .git_local_tools import (
    RenameFileHandler,
    CreateSymlinkHandler,
    ModifyFilesLocalHandler,
    CreateBranchLocalHandler,
    GetBranchesLocalHandler,
    GetDiffLocalHandler
)


def create_default_tool_executor() -> ToolExecutor:
    """
    Create a ToolExecutor with all default tools registered.

    Returns:
        ToolExecutor with all standard tools
    """
    executor = ToolExecutor()

    # Register GitHub tools
    executor.register(ReadIssueHandler())
    executor.register(CreateBranchHandler())
    executor.register(ModifyFileHandler())
    executor.register(CommitChangesHandler())
    executor.register(CreatePullRequestHandler())
    executor.register(PostCommentHandler())

    # Register file tools
    executor.register(ReadFileHandler())
    executor.register(ListFilesHandler())
    executor.register(SearchFilesHandler())
    executor.register(GetRepositoryOverviewHandler())

    # Register code search tools
    executor.register(SearchCodeHandler())
    executor.register(FindSymbolHandler())
    executor.register(GrepPatternHandler())

    # Register discovery tools
    executor.register(DiscoverRelevantFilesHandler())

    # Register task tools
    executor.register(AttemptCompletionHandler())
    # NOTE: ask_followup_question is disabled for webhook-based execution
    # It's designed for interactive CLI sessions where a user can respond
    # In webhook mode, the agent just loops infinitely waiting for answers that never come
    # ask_followup_handler = AskFollowupQuestionHandler()
    # executor.register(ask_followup_handler)
    executor.register(CreatePlanHandler())

    # Register validation tools (no user interaction in webhook mode)
    executor.register(RunValidationHandler(ask_followup_handler=None))

    # Register local git operation tools
    executor.register(RenameFileHandler())
    executor.register(CreateSymlinkHandler())
    executor.register(ModifyFilesLocalHandler())
    executor.register(CreateBranchLocalHandler())
    executor.register(GetBranchesLocalHandler())
    executor.register(GetDiffLocalHandler())

    # Total: 23 tools (17 previous + 6 local git tools)
    return executor


__all__ = [
    # Base classes
    "IToolHandler",
    "BaseToolHandler",
    "ToolDefinition",
    "ToolResponse",
    "ToolCategory",
    # Executor
    "ToolExecutor",
    "create_default_tool_executor",
    # GitHub tools
    "ReadIssueHandler",
    "CreateBranchHandler",
    "ModifyFileHandler",
    "CommitChangesHandler",
    "CreatePullRequestHandler",
    "PostCommentHandler",
    # File tools
    "ReadFileHandler",
    "ListFilesHandler",
    "SearchFilesHandler",
    "GetRepositoryOverviewHandler",
    # Code search tools
    "SearchCodeHandler",
    "FindSymbolHandler",
    "GrepPatternHandler",
    # Discovery tools
    "DiscoverRelevantFilesHandler",
    # Task tools
    "AttemptCompletionHandler",
    "AskFollowupQuestionHandler",
    "CreatePlanHandler",
    # Validation tools
    "RunValidationHandler",
    # Local git tools
    "RenameFileHandler",
    "CreateSymlinkHandler",
    "ModifyFilesLocalHandler",
    "CreateBranchLocalHandler",
    "GetBranchesLocalHandler",
    "GetDiffLocalHandler",
]
