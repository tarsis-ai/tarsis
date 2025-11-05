"""
Google Gemini LLM Provider implementation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

try:
    import google.generativeai as genai
except ImportError:
    genai = None

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
                "google-generativeai package is required for Gemini provider. "
                "Install with: pip install google-generativeai"
            )

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required (set GEMINI_API_KEY env var)")

        # Configure the SDK
        genai.configure(api_key=self.api_key)

        # Initialize the model
        self.model = genai.GenerativeModel(model_name=self.model_id)

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

        # Create model with system instruction and tools
        model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=system_prompt if system_prompt else None,
            tools=gemini_tools
        )

        # Configure generation
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        logger.debug(f"Creating Gemini message with {len(gemini_messages)} messages, "
                    f"max_tokens={max_tokens}, tools={'enabled' if gemini_tools else 'disabled'}")

        # Generate content
        response = model.generate_content(
            gemini_messages,
            generation_config=generation_config
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

        # Create model with system instruction and tools
        model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=system_prompt if system_prompt else None,
            tools=gemini_tools
        )

        # Configure generation
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        # Stream response
        response = model.generate_content(
            gemini_messages,
            generation_config=generation_config,
            stream=True
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

                        elif block_type == "tool_result":
                            # Convert tool result to function response
                            tool_use_id = block.get("tool_use_id", "")
                            result_content = block.get("content", "")

                            # Gemini expects function responses in a specific format
                            parts.append({
                                "function_response": {
                                    "name": tool_use_id,  # Use tool_use_id as name
                                    "response": {"result": result_content}
                                }
                            })

            if parts:
                formatted.append({
                    "role": gemini_role,
                    "parts": parts
                })

        return formatted

    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Anthropic tool format to Gemini format.

        Anthropic format:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": {...}
        }

        Gemini format:
        {
            "name": "tool_name",
            "description": "...",
            "parameters": {...}
        }
        """
        gemini_tools = []

        for tool in tools:
            gemini_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]  # Just rename the field
            })

        logger.debug(f"Converted {len(tools)} tools to Gemini format")

        return gemini_tools

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
        except ValueError:
            # response.text raises ValueError if there's no text (e.g., only function calls)
            pass

        # Extract function calls (tool uses)
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for i, part in enumerate(candidate.content.parts):
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        # Convert to Anthropic tool_use format
                        content_blocks.append({
                            "type": "tool_use",
                            "id": f"tool_{i}",  # Generate ID
                            "name": fc.name,
                            "input": dict(fc.args) if fc.args else {}
                        })

        # If no content blocks, return empty text
        if not content_blocks:
            content_blocks = [{"type": "text", "text": ""}]

        # Log parsed content
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
        if tool_uses:
            logger.debug(f"Parsed {len(tool_uses)} tool use(s) from response")
        else:
            logger.debug(f"Parsed text response (no tool uses)")

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
            if hasattr(candidate, 'finish_reason'):
                # Map Gemini finish reasons to Anthropic-like reasons
                finish_reason = str(candidate.finish_reason)
                if 'STOP' in finish_reason:
                    stop_reason = "end_turn"
                elif 'MAX_TOKENS' in finish_reason:
                    stop_reason = "max_tokens"

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
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
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
