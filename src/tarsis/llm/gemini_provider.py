"""
Google Gemini LLM Provider implementation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from .provider import (
    ILLMProvider,
    BaseLLMProvider,
    ModelInfo,
    ModelProvider,
    AssistantMessage,
    Usage
)
from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Provider for Google Gemini models.

    Supports:
    - Gemini 2.5 Pro (1M token context)
    - Gemini 2.5 Flash (1M token context)
    - Gemini 2.5 Flash Lite (1M token context)
    - Tool calling
    - Streaming
    """

    # Model definitions
    MODELS = {
        "gemini-2.5-pro": ModelInfo(
            id="gemini-2.5-pro",
            name="Gemini 2.5 Pro",
            provider=ModelProvider.GEMINI,
            context_window=1048576,  # 1M input tokens
            supports_tools=True,
            supports_streaming=True
        ),
        "gemini-2.5-flash": ModelInfo(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            provider=ModelProvider.GEMINI,
            context_window=1048576,  # 1M input tokens
            supports_tools=True,
            supports_streaming=True
        ),
        "gemini-2.5-flash-lite": ModelInfo(
            id="gemini-2.5-flash-lite",
            name="Gemini 2.5 Flash Lite",
            provider=ModelProvider.GEMINI,
            context_window=1048576,  # 1M input tokens
            supports_tools=True,
            supports_streaming=True
        )
    }

    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize Gemini provider.

        Args:
            model_id: Gemini model ID
            api_key: Google API key (or from GEMINI_API_KEY env var)
        """
        super().__init__(model_id, api_key)

        if genai is None:
            raise ImportError(
                "google-genai package is required for Gemini provider. "
                "Install with: pip install google-genai"
            )

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required (set GEMINI_API_KEY env var)")

        # Initialize the client with the new API
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"Gemini provider initialized with model: {self.model_id}")

    @property
    def model_info(self) -> ModelInfo:
        """Get model information"""
        return self.MODELS.get(
            self.model_id,
            # Default fallback
            ModelInfo(
                id=self.model_id,
                name=self.model_id,
                provider=ModelProvider.GEMINI,
                context_window=1048576,  # Assume 1M default
                supports_tools=True,
                supports_streaming=True
            )
        )

    @retry_with_backoff(max_retries=3)
    async def create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> AssistantMessage:
        """
        Create a message using Gemini API.

        Args:
            system_prompt: System prompt
            messages: Conversation history
            tools: Available tools (in Anthropic format)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            AssistantMessage with response
        """
        # Convert messages to Gemini format
        gemini_messages = self._format_messages_for_gemini(messages)

        # Convert tools if provided
        gemini_tools = None
        if tools:
            gemini_tools = self._convert_tools_to_gemini_format(tools)

        # Build configuration using types.GenerateContentConfig
        config_params = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }

        # Add system instruction if provided
        if system_prompt:
            config_params['system_instruction'] = system_prompt

        # Add tools if provided
        if gemini_tools:
            config_params['tools'] = gemini_tools
            # Disable automatic function calling
            config_params['automatic_function_calling'] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        # Create typed config object
        config = types.GenerateContentConfig(**config_params)

        logger.debug(f"Creating Gemini message with {len(gemini_messages)} messages, "
                    f"max_tokens={max_tokens}, tools={'enabled' if gemini_tools else 'disabled'}")

        # Generate content using the new API
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=gemini_messages,
            config=config
        )

        # Log response details
        if hasattr(response, 'usage_metadata'):
            logger.debug(f"Token usage: input={response.usage_metadata.prompt_token_count}, "
                        f"output={response.usage_metadata.candidates_token_count}")

        logger.debug(f"Response received with {len(response.candidates) if hasattr(response, 'candidates') else 0} candidates")

        # Parse response
        return self._parse_gemini_response(response)

    async def create_message_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Create a streaming message using Gemini.

        Yields stream events.
        """
        # Convert messages to Gemini format
        gemini_messages = self._format_messages_for_gemini(messages)

        # Convert tools if provided
        gemini_tools = None
        if tools:
            gemini_tools = self._convert_tools_to_gemini_format(tools)

        # Build configuration using types.GenerateContentConfig
        config_params = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }

        # Add system instruction if provided
        if system_prompt:
            config_params['system_instruction'] = system_prompt

        # Add tools if provided
        if gemini_tools:
            config_params['tools'] = gemini_tools
            # Disable automatic function calling
            config_params['automatic_function_calling'] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        # Create typed config object
        config = types.GenerateContentConfig(**config_params)

        # Stream response using the new API
        response = self.client.models.generate_content_stream(
            model=self.model_id,
            contents=gemini_messages,
            config=config
        )

        for chunk in response:
            yield self._parse_stream_chunk(chunk)

    def _format_messages_for_gemini(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert Anthropic-style messages to Gemini format.

        Anthropic format:
        {
            "role": "user" | "assistant",
            "content": string | [{"type": "text", "text": "..."}, ...]
        }

        Gemini format:
        {
            "role": "user" | "model",
            "parts": [{"text": "..."}, ...]
        }
        """
        formatted = []
        # Track mapping from tool_use_id to function name for function responses
        tool_use_id_to_name = {}

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Map role
            gemini_role = "model" if role == "assistant" else role

            # Convert content to parts
            parts = []

            if isinstance(content, str):
                # Simple text content
                parts.append({"text": content})
            elif isinstance(content, list):
                # Content blocks
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        if block_type == "text":
                            parts.append({"text": block.get("text", "")})

                        elif block_type == "tool_use":
                            # Track tool_use blocks from assistant messages
                            # We need this mapping to correctly format function responses later
                            tool_use_id = block.get("id", "")
                            function_name = block.get("name", "")
                            if tool_use_id and function_name:
                                tool_use_id_to_name[tool_use_id] = function_name

                        elif block_type == "tool_result":
                            # Convert tool result to function response
                            tool_use_id = block.get("tool_use_id", "")
                            result_content = block.get("content", "")

                            # Look up the actual function name from our mapping
                            function_name = tool_use_id_to_name.get(tool_use_id, tool_use_id)

                            if not function_name:
                                logger.warning(f"Could not find function name for tool_use_id: {tool_use_id}")
                                continue

                            # Use typed Part object for function response
                            function_response_part = types.Part.from_function_response(
                                name=function_name,
                                response={'result': result_content}
                            )
                            parts.append(function_response_part)

            if parts:
                formatted.append({
                    "role": gemini_role,
                    "parts": parts
                })

        return formatted

    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert Anthropic tool format to Gemini format.

        Anthropic format:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": {...}
        }

        New Gemini API format:
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="tool_name",
                description="...",
                parameters_json_schema={...}  # Standard JSON schema
            )
        ])
        """
        function_declarations = []

        for tool in tools:
            # Use standard JSON schema (already in correct format from Anthropic)
            input_schema = tool["input_schema"]

            function_declaration = types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters_json_schema=input_schema
            )
            function_declarations.append(function_declaration)

        # Wrap all function declarations in a single Tool object
        gemini_tool = types.Tool(function_declarations=function_declarations)

        logger.debug(f"Converted {len(tools)} tools to Gemini format")

        return [gemini_tool]  # Return as list containing single Tool object

    def _parse_gemini_response(self, response: Any) -> AssistantMessage:
        """
        Parse Gemini response into AssistantMessage.

        Converts Gemini format to Anthropic-compatible format.
        """
        content_blocks = []

        # Extract text content
        try:
            if hasattr(response, 'text') and response.text:
                content_blocks.append({"type": "text", "text": response.text})
        except ValueError as e:
            # response.text raises ValueError if there's no text (e.g., only function calls)
            logger.debug(f"No text in response: {e}")

        # Track if we encountered malformed function calls
        malformed_calls = []

        # Debugging: check if we have candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            logger.warning("Response has no candidates")

        # Extract function calls (tool uses)
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            # Debugging: check candidate structure
            if not hasattr(candidate, 'content'):
                logger.warning("Candidate has no content attribute")
            elif not hasattr(candidate.content, 'parts'):
                logger.warning("Candidate content has no parts attribute")
            elif candidate.content.parts is None:
                logger.warning("Candidate content parts is None")

            if (hasattr(candidate, 'content') and
                hasattr(candidate.content, 'parts') and
                candidate.content.parts is not None):

                # Log if we have no parts at all
                if not candidate.content.parts:
                    logger.warning("Response has content but no parts")

                tool_call_index = 0
                for part in candidate.content.parts:
                    # Handle function calls
                    if hasattr(part, 'function_call'):
                        fc = part.function_call

                        # Validate function call has a non-empty name
                        if not hasattr(fc, 'name') or not fc.name or fc.name.strip() == "":
                            # Log detailed info about malformed call for debugging
                            args_info = dict(fc.args) if hasattr(fc, 'args') and fc.args else {}
                            logger.warning(
                                f"Skipping malformed function call with empty name. "
                                f"Part type: {type(part).__name__}, "
                                f"Has args: {hasattr(fc, 'args')}, "
                                f"Args: {args_info}"
                            )
                            malformed_calls.append(args_info)
                            continue

                        # Convert to Anthropic tool_use format
                        content_blocks.append({
                            "type": "tool_use",
                            "id": f"tool_{tool_call_index}",  # Generate ID
                            "name": fc.name,
                            "input": dict(fc.args) if fc.args else {}
                        })
                        tool_call_index += 1

                    # Handle text parts (including thought_signature and other text-like parts)
                    elif hasattr(part, 'text') and part.text:
                        # Only add if we haven't already added text via response.text
                        if not any(b.get('type') == 'text' for b in content_blocks):
                            content_blocks.append({"type": "text", "text": part.text})

                    # Log unexpected part types for debugging
                    elif not hasattr(part, 'function_call'):
                        part_type = type(part).__name__
                        logger.debug(f"Encountered non-function-call part: {part_type}")

        # If no content blocks but we had malformed calls, add error message
        if not content_blocks and malformed_calls:
            error_msg = (
                "I encountered an issue with my response. "
                "I tried to call a function but the call was malformed. "
                "Let me try a different approach."
            )
            logger.error(
                f"All function calls were malformed, returning error message. "
                f"Malformed calls: {len(malformed_calls)}"
            )
            content_blocks = [{"type": "text", "text": error_msg}]
        # If no content blocks at all, return empty text
        elif not content_blocks:
            content_blocks = [{"type": "text", "text": ""}]

        # Log parsed content
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
        if tool_uses:
            logger.debug(f"Parsed {len(tool_uses)} tool use(s) from response")
        else:
            logger.debug(f"Parsed text response (no tool uses)")

        # Log warning if we skipped malformed calls but still have valid content
        if malformed_calls and tool_uses:
            logger.warning(
                f"Skipped {len(malformed_calls)} malformed function call(s) "
                f"but successfully parsed {len(tool_uses)} valid tool use(s)"
            )

        # Extract usage metadata
        usage = None
        if hasattr(response, 'usage_metadata'):
            um = response.usage_metadata
            usage = Usage(
                input_tokens=getattr(um, 'prompt_token_count', 0),
                output_tokens=getattr(um, 'candidates_token_count', 0),
                total_tokens=getattr(um, 'total_token_count', 0)
            )

        # Determine stop reason
        stop_reason = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            # Log safety ratings if available for debugging
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                logger.debug(f"Safety ratings: {candidate.safety_ratings}")

            if hasattr(candidate, 'finish_reason'):
                # Map Gemini finish reasons to Anthropic-like reasons
                finish_reason = str(candidate.finish_reason)
                logger.debug(f"Gemini finish_reason: {finish_reason}")

                if 'STOP' in finish_reason:
                    stop_reason = "end_turn"
                elif 'MAX_TOKENS' in finish_reason:
                    stop_reason = "max_tokens"
                elif 'SAFETY' in finish_reason:
                    logger.warning(
                        f"Response stopped due to safety filters: {finish_reason}. "
                        f"Safety ratings: {getattr(candidate, 'safety_ratings', 'N/A')}"
                    )
                    stop_reason = "stop_sequence"
                elif 'RECITATION' in finish_reason:
                    logger.warning(f"Response stopped due to recitation: {finish_reason}")
                    stop_reason = "stop_sequence"
                elif 'OTHER' in finish_reason:
                    logger.warning(f"Response stopped for unspecified reason: {finish_reason}")
                else:
                    logger.warning(f"Unknown finish_reason: {finish_reason}")
            else:
                logger.warning("Candidate has no finish_reason attribute")

        return AssistantMessage(
            content=content_blocks,
            stop_reason=stop_reason,
            usage=usage
        )

    def _parse_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """
        Parse a streaming chunk.

        Returns a dictionary with chunk information.
        """
        chunk_data = {
            "type": "chunk",
            "data": {}
        }

        # Extract text if available
        try:
            if hasattr(chunk, 'text') and chunk.text:
                chunk_data["data"]["text"] = chunk.text
        except ValueError:
            pass

        # Extract function calls if available
        if hasattr(chunk, 'candidates') and chunk.candidates:
            candidate = chunk.candidates[0]
            if (hasattr(candidate, 'content') and
                hasattr(candidate.content, 'parts') and
                candidate.content.parts is not None):
                function_calls = []
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        function_calls.append({
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {}
                        })
                if function_calls:
                    chunk_data["data"]["function_calls"] = function_calls

        return chunk_data
