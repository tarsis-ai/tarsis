"""
Reflection Prompt Templates for the Reflexion Framework

Contains context-specific prompts for different types of reflection triggers:
- Validation failures (tests, linting, static analysis)
- Tool execution errors
- Consecutive mistakes
- Periodic checkpoints
- Trial failures (multi-trial mode)

Each prompt guides the LLM to generate actionable insights for improvement.
"""

# ==============================================================================
# System Prompt for Reflection
# ==============================================================================

REFLECTION_SYSTEM_PROMPT = """You are an AI agent reflecting on your own performance to learn from mistakes.

Your goal is to analyze what went wrong and extract concrete, actionable lessons for future attempts.

Be brutally honest about your mistakes. Focus on:
1. **Root causes** - What fundamental error did you make?
2. **Patterns** - Have you made similar mistakes before?
3. **Concrete fixes** - What specific actions will prevent this?
4. **General lessons** - What should you remember long-term?

Your reflections will guide your future decision-making. Be specific and actionable.

Format your reflection clearly with these sections:
- **Root Cause**: What specifically went wrong
- **Why It Happened**: Your reasoning at the time
- **Pattern Recognition**: Similar past mistakes (if any)
- **Concrete Fix**: Exact steps to resolve this
- **Learning**: General lesson to remember
"""

# ==============================================================================
# Validation Failure Prompt
# ==============================================================================

VALIDATION_FAILURE_PROMPT = """You just ran validation on your code changes but encountered failures.

VALIDATION RESULTS:
{validation_summary}

FAILED TESTS:
{failed_tests}

LINTING ISSUES:
{lint_issues}

STATIC ANALYSIS ERRORS:
{static_errors}

YOUR RECENT ACTIONS:
{recent_actions}

PREVIOUS REFLECTIONS:
{previous_reflections}

Reflect deeply on what went wrong:

1. **Root Cause**: What specific code changes caused these failures? Be precise about which files and functions.

2. **Why It Happened**: Why did you make those choices at the time? What did you misunderstand about the requirements or codebase?

3. **Pattern Recognition**: Have you made similar mistakes before in this task? Look at your previous reflections.

4. **Concrete Fix**: What exact changes will fix this? Be specific:
   - Which files need modification?
   - What code needs to change?
   - Which tests need updating?

5. **Learning**: What should you remember for the rest of this task and future tasks? What pattern should you watch for?

Be specific, actionable, and honest about your mistakes.

REFLECTION:
"""

# ==============================================================================
# Tool Error Prompt
# ==============================================================================

TOOL_ERROR_PROMPT = """You just attempted to use the '{tool_name}' tool but it failed with an error.

ERROR: {error_message}

ERROR TYPE: {error_type}

TOOL INPUT:
{tool_input}

YOUR RECENT ACTIONS:
{recent_actions}

PREVIOUS REFLECTIONS:
{previous_reflections}

Reflect on what went wrong:

1. **What Happened**: Why did this tool fail? Analyze the error message carefully.

2. **Your Assumption**: What did you assume about the tool/file/state that was wrong?
   - Did you assume a file existed that doesn't?
   - Did you assume a branch was created that wasn't?
   - Did you misunderstand the tool's requirements?

3. **Missing Knowledge**: What information should you have gathered first before using this tool?
   - Should you have listed files first?
   - Should you have checked branch status?
   - Should you have read documentation?

4. **Correct Approach**: What should you do instead? Be specific about the sequence of actions.

5. **Prevention**: How can you avoid this error in future tool calls? What pattern should you follow?

REFLECTION:
"""

# ==============================================================================
# Consecutive Mistakes Prompt
# ==============================================================================

CONSECUTIVE_MISTAKES_PROMPT = """You have made {mistake_count} consecutive mistakes and are at risk of aborting the task.

RECENT ERRORS:
{recent_errors}

PATTERN DETECTED: You are repeatedly making mistakes without learning.

PREVIOUS REFLECTIONS:
{previous_reflections}

This is a critical moment. Reflect deeply:

1. **Pattern Analysis**: What pattern do you see in your mistakes? Are you:
   - Repeating the same error?
   - Making assumptions without verification?
   - Misunderstanding the requirements?
   - Using tools incorrectly?

2. **Fundamental Issue**: What fundamental misunderstanding is causing repeated failures?
   - Do you misunderstand the codebase structure?
   - Are you missing key context about the task?
   - Are you using the wrong approach entirely?

3. **Strategy Change**: Your current approach isn't working. What completely different strategy should you try?
   - Should you explore the codebase more first?
   - Should you read more documentation?
   - Should you break the problem down differently?
   - Should you use different tools?

4. **Knowledge Gap**: What key information about this codebase/task are you missing? How can you acquire it?

5. **Action Plan**: What are the next 3 concrete steps you should take to break this pattern?
   Step 1:
   Step 2:
   Step 3:

Be honest about what's not working and propose a radically different approach.

REFLECTION:
"""

# ==============================================================================
# Periodic Checkpoint Prompt
# ==============================================================================

PERIODIC_CHECKPOINT_PROMPT = """You have completed {iteration} iterations. Time for a progress review.

PROGRESS SO FAR:
- Files accessed: {files_accessed}
- Files modified: {files_modified}
- Validation performed: {validation_performed}
- Validation passed: {validation_passed}

TOOLS USED:
{tools_used}

PREVIOUS REFLECTIONS:
{previous_reflections}

Reflect on your progress:

1. **Progress Assessment**: Are you making good progress toward the goal?
   - Have you modified the right files?
   - Are you on track to complete the task?
   - Are you stuck or moving forward?

2. **Strategy Effectiveness**: Is your current approach working?
   - Are the tools you're using appropriate?
   - Is your understanding of the task correct?
   - Should you adjust your strategy?

3. **Obstacles**: What obstacles or challenges have you encountered?
   - Technical difficulties?
   - Missing information?
   - Unclear requirements?

4. **Adjustments**: What adjustments should you make for the next phase?
   - Different approach?
   - Additional exploration?
   - Different tools?

5. **Missing Steps**: What important steps might you be overlooking?
   - Validation?
   - Testing?
   - Documentation?
   - Error handling?

Be honest about whether you're on the right track.

REFLECTION:
"""

# ==============================================================================
# Trial Failure Prompt (Multi-Trial Mode)
# ==============================================================================

TRIAL_FAILURE_PROMPT = """Trial {trial_number} has ended without successfully completing the task.

TRIAL SUMMARY:
- Iterations used: {iterations_used}
- Files modified: {files_modified}
- Validation performed: {validation_performed}
- Validation passed: {validation_passed}
- Abort reason: {abort_reason}
- Completion attempted: {completion_attempted}

TOOLS USED:
{tools_used}

KEY DECISIONS MADE:
{key_decisions}

FULL CONVERSATION SUMMARY:
{full_conversation}

PREVIOUS REFLECTIONS (from earlier trials):
{previous_reflections}

This trial failed. Reflect on the entire attempt:

1. **What Went Wrong**: What was the fundamental flaw in this trial's approach?
   - Wrong strategy?
   - Missing information?
   - Technical errors?
   - Misunderstood requirements?

2. **Critical Mistakes**: What were the 2-3 most critical mistakes in this trial?
   For each mistake:
   - What was the mistake?
   - When did it occur (which iteration)?
   - How did it impact the rest of the trial?

3. **What Worked**: What aspects of this trial worked well and should be kept?
   - Good decisions?
   - Successful steps?
   - Correct understanding?

4. **Learning from Past Trials**: Looking at reflections from previous trials, what patterns do you see?
   - Are you repeating the same mistakes?
   - Have you addressed previous issues?
   - What new problems emerged?

5. **Next Trial Strategy**: For the next trial, what completely different approach will you take?
   - What will you do first?
   - What will you avoid?
   - What new information will you gather?
   - What assumptions will you verify?

This is your chance to learn and improve for the next trial. Be thorough and honest.

REFLECTION:
"""

# ==============================================================================
# Pre-Completion Verification Prompt
# ==============================================================================

PRE_COMPLETION_PROMPT = """You are about to mark this task as complete. Before doing so, carefully verify that ALL requirements have been met.

ORIGINAL TASK/ISSUE:
{original_task}

YOUR WORK SO FAR:
- Iterations used: {iterations_used}
- Files created/modified: {files_modified}
- Validation performed: {validation_performed}
- Validation passed: {validation_passed}

TOOLS USED:
{tools_used}

FILES YOU CREATED/MODIFIED:
{modified_files_list}

CRITICAL PRE-COMPLETION VERIFICATION:

1. **Requirements Checklist**: Go through the original task/issue line by line:
   - List each requirement or deliverable mentioned
   - For each one, verify if it has been completed
   - Identify any requirements that are MISSING or INCOMPLETE

2. **File Verification**:
   - Does the task require creating multiple files?
   - Have you created ALL required files?
   - Are any files mentioned in the requirements but NOT in your modified files list?

3. **Test Coverage**:
   - Does the task require tests?
   - Have you created test files?
   - Are the tests comprehensive?

4. **Documentation**:
   - Does the task require documentation?
   - Have you added necessary docs/comments?

5. **Validation Status**:
   - Did validation pass?
   - If no validation was run, should it have been?
   - Are there any uncaught errors?

DECISION:
Based on this verification, is the task TRULY complete, or are there missing requirements?

If INCOMPLETE:
- List exactly what is missing
- Explain why it was missed
- Specify what needs to be done next

If COMPLETE:
- Confirm that ALL requirements have been verified
- List what was accomplished

VERIFICATION REFLECTION:
"""

# ==============================================================================
# Helper Functions
# ==============================================================================

def format_validation_summary(validation_result: dict) -> str:
    """Format validation result for reflection prompt"""
    if not validation_result:
        return "No validation result available"

    status = validation_result.get("validation_status", "unknown")
    tier = validation_result.get("tier_used", "unknown")
    passed = validation_result.get("passed", False)

    summary = f"Status: {'PASSED' if passed else 'FAILED'}\n"
    summary += f"Validation Tier: {tier}\n"

    if not passed and "failure_summary" in validation_result:
        summary += f"\nFailure Summary:\n{validation_result['failure_summary']}"

    return summary


def format_failed_tests(validation_result: dict) -> str:
    """Extract failed tests from validation result"""
    if not validation_result or validation_result.get("passed", True):
        return "No failed tests"

    failure_summary = validation_result.get("failure_summary", "")
    if "test" in failure_summary.lower():
        return failure_summary

    return "Test failure details not available"


def format_lint_issues(validation_result: dict) -> str:
    """Extract linting issues from validation result"""
    if not validation_result:
        return "No linting issues"

    tier = validation_result.get("tier_used", "")
    if "lint" not in tier.lower():
        return "Linting not performed"

    failure_summary = validation_result.get("failure_summary", "")
    if "lint" in failure_summary.lower() or "style" in failure_summary.lower():
        return failure_summary

    return "No linting issues found"


def format_static_errors(validation_result: dict) -> str:
    """Extract static analysis errors from validation result"""
    if not validation_result:
        return "No static analysis errors"

    tier = validation_result.get("tier_used", "")
    if "static" not in tier.lower() and "type" not in tier.lower():
        return "Static analysis not performed"

    failure_summary = validation_result.get("failure_summary", "")
    if failure_summary and not failure_summary.startswith("No"):
        return failure_summary

    return "No static analysis errors found"


def format_tools_used(tools_used: dict) -> str:
    """Format tool usage counts for reflection prompt"""
    if not tools_used:
        return "No tools used yet"

    formatted = ""
    for tool_name, count in sorted(tools_used.items(), key=lambda x: x[1], reverse=True):
        formatted += f"- {tool_name}: {count} times\n"

    return formatted.strip()
