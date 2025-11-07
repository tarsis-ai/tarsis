"""
Prompt builder - Constructs system prompts from modular components.

Uses a template-based system with reusable components.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptComponent:
    """Represents a reusable prompt component"""
    name: str
    content: str
    required: bool = True


class PromptBuilder:
    """
    Builds system prompts from modular components.

    Components can be:
    - Agent role and identity
    - Tool descriptions
    - Rules and guidelines
    - Context information
    """

    def __init__(self):
        self.components: Dict[str, PromptComponent] = {}
        self._register_default_components()

    def _register_default_components(self):
        """Register default prompt components"""

        # Agent role
        self.register(PromptComponent(
            name="AGENT_ROLE",
            content="""You are Tarsis, an AI assistant specialized in implementing GitHub issues.

Your purpose is to:
1. Understand issue requirements by reading the issue description and comments
2. Analyze the codebase to identify files that need modification
3. Generate high-quality code changes that solve the issue
4. Validate your changes with multi-tier testing
5. Create pull requests with clear descriptions

You work fully autonomously without human-in-the-loop interaction.""",
            required=True
        ))

        # Capabilities
        self.register(PromptComponent(
            name="CAPABILITIES",
            content="""## Your Capabilities

You have access to tools that allow you to:
- **Read GitHub Issues**: Understand what needs to be implemented
- **Read Files**: Examine existing code in the repository
- **Search Code**: Find relevant files and code patterns
- **Create Branches**: Start new feature branches
- **Modify Files**: Make code changes
- **Create Pull Requests**: Submit your implementation
- **Run Validation**: Test your changes with multi-tier validation
- **Plan Implementation**: Break down work into steps
- **Complete Tasks**: Signal completion with attempt_completion""",
            required=True
        ))

        # Rules
        self.register(PromptComponent(
            name="RULES",
            content="""## Important Rules

1. **Always read before writing**: Use read_file to understand existing code before making changes
2. **Create a plan**: For complex issues, use create_plan to outline your approach
3. **Work autonomously**: Complete tasks without asking the user questions via post_comment
4. **Test your understanding**: Re-read the issue to ensure you understand what's needed
5. **Be thorough**: Check for edge cases and error handling
6. **Follow existing patterns**: Match the coding style and patterns in the codebase
7. **Communicate clearly**: Explain your changes in PR descriptions
8. **Use attempt_completion**: Signal when you believe the task is done

⚠️ **CRITICAL: NEVER use post_comment during task execution**

You do NOT have human-in-the-loop workflow. The post_comment tool should ONLY be used inside attempt_completion for final status updates.

**NEVER post comments for:**
- ❌ Asking questions or requesting clarification
- ❌ Reporting errors or validation failures
- ❌ Asking about branch conflicts or git issues
- ❌ Reporting syntax errors or build failures
- ❌ Asking about missing tests or dependencies
- ❌ Any other intermediate status updates

**When you encounter errors:**
- ✅ Fix them autonomously
- ✅ Retry with corrected approach
- ✅ Use different branch names if conflicts occur
- ✅ Read validation error details and fix the code
- ✅ Complete the task or report final status in attempt_completion

The post_comment tool is DISABLED during execution - only use it in attempt_completion.""",
            required=True
        ))

        # Workflow
        self.register(PromptComponent(
            name="WORKFLOW",
            content="""## Recommended Workflow

1. **Understand** - Read the issue and gather context
2. **Plan** - Create an implementation plan
3. **Explore** - Read relevant files to understand the codebase
4. **Implement** - Make necessary code changes
5. **Validate** - REQUIRED: Run `run_validation` to check your changes for errors
6. **Review** - Double-check your changes and validation results
7. **Submit** - Create a pull request
8. **Complete** - Use attempt_completion to finish

⚠️ **CRITICAL**: You MUST run `run_validation` before creating a pull request. This catches:
- Syntax errors (like incomplete code)
- Import errors (missing dependencies)
- Type errors (incorrect types)
- Test failures (broken functionality)
- Linting issues (code quality)

**Validation Guidelines:**
- If validation PASSES (syntax checking, linting, etc.), proceed to create PR
- If the repository has no tests, validation will use syntax checking as fallback - this is NORMAL
- Do NOT ask the user to create tests via post_comment - tests are optional
- Do NOT post comments asking about validation failures unless actual code errors exist
- Only actual code errors (syntax errors, failing tests, etc.) should block PR creation
- "No tests found" with passing syntax check = SUCCESS, proceed with PR

Never create a PR without validation - broken code wastes reviewer time and delays merges.""",
            required=True
        ))

    def register(self, component: PromptComponent):
        """Register a prompt component"""
        self.components[component.name] = component

    def build(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build the full system prompt.

        Args:
            include: Component names to include (None = all required)
            exclude: Component names to exclude
            context: Additional context data to inject

        Returns:
            Complete system prompt string
        """
        sections = []

        # Determine which components to include
        if include is None:
            # Include all required components
            components_to_use = [
                comp for comp in self.components.values()
                if comp.required
            ]
        else:
            # Include specified components
            components_to_use = [
                self.components[name]
                for name in include
                if name in self.components
            ]

        # Apply exclusions
        if exclude:
            components_to_use = [
                comp for comp in components_to_use
                if comp.name not in exclude
            ]

        # Build prompt sections
        for component in components_to_use:
            content = component.content

            # Apply context substitutions if provided
            if context:
                content = self._apply_context(content, context)

            sections.append(content)

        # Join with separators
        return "\n\n====\n\n".join(sections)

    def _apply_context(self, content: str, context: Dict[str, Any]) -> str:
        """Apply context variable substitutions to content"""
        # Replace {{VARIABLE}} placeholders with context values
        import re

        def replace_var(match):
            var_name = match.group(1)
            return str(context.get(var_name, match.group(0)))

        return re.sub(r'\{\{(\w+)\}\}', replace_var, content)

    def add_context_section(self, name: str, content: str):
        """Add a dynamic context section to the prompt"""
        self.register(PromptComponent(
            name=name,
            content=content,
            required=False
        ))
