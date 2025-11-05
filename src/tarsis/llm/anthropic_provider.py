"""
Anthropic (Claude) LLM Provider implementation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
import anthropic

logger = logging.getLogger(__name__)

from .provider import (
    ILLMProvider,
    BaseLLMProvider,
    ModelInfo,
    ModelProvider,
    AssistantMessage,
    Usage
)
from ..utils.retry import retry_with_backoff


class AnthropicProvider(BaseLLMProvider):
    """
    Provider for Anthropic's Claude models.

    Supports:
    - Claude 3.5 Sonnet
    - Claude 3 Opus
    - Claude 3 Haiku
    - Tool calling
    - Streaming
    """

    # Model definitions
    MODELS = {
        "claude-3-5-sonnet-20241022": ModelInfo(
            id="claude-3-5-sonnet-20241022",
            name="Claude 3.5 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            context_window=200000,
            supports_tools=True,
            supports_streaming=True
        ),
        "claude-3-opus-20240229": ModelInfo(
            id="claude-3-opus-20240229",
            name="Claude 3 Opus",
            provider=ModelProvider.ANTHROPIC,
            context_window=200000,
            supports_tools=True,
            supports_streaming=True
        ),
        "claude-3-haiku-20240307": ModelInfo(
            id="claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            provider=ModelProvider.ANTHROPIC,
            context_window=200000,
            supports_tools=True,
            supports_streaming=True
        )
    }

    def __init__(self, model_id: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.

        Args:
            model_id: Claude model ID
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_id, api_key)

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY env var)")

        # Initialize client
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

        logger.info(f"Anthropic provider initialized with model: {self.model_id}")

    @property
    def model_info(self) -> ModelInfo:
        """Get model information"""
        return self.MODELS.get(
            self.model_id,
            # Default fallback
            ModelInfo(
                id=self.model_id,
                name=self.model_id,
                provider=ModelProvider.ANTHROPIC,
                context_window=200000,
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
        Create a message using Anthropic API.

        Args:
            system_prompt: System prompt
            messages: Conversation history
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            AssistantMessage with response
        """
        # Build request parameters
        params = {
            "model": self.model_id,
            "system": system_prompt,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools
            logger.debug(f"Request includes {len(tools)} tools")

        logger.debug(f"Creating message with {len(messages)} messages, max_tokens={max_tokens}")

        # Make API call
        response = await self.client.messages.create(**params)

        # Log response details
        logger.debug(f"Response received: stop_reason={response.stop_reason}")
        if hasattr(response, 'usage'):
            logger.debug(f"Token usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

        # Parse response
        return self._parse_response(response)

    async def create_message_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Create a streaming message.

        Yields stream events.
        """
        # Build request parameters
        params = {
            "model": self.model_id,
            "system": system_prompt,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools

        # Stream response
        async with self.client.messages.stream(**params) as stream:
            async for event in stream:
                yield self._parse_stream_event(event)

    def _parse_response(self, response: Any) -> AssistantMessage:
        """Parse Anthropic API response into AssistantMessage"""
        # Extract usage
        usage = None
        if hasattr(response, 'usage'):
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )

        # Content is already in the right format (list of content blocks)
        return AssistantMessage(
            content=response.content,
            stop_reason=response.stop_reason,
            usage=usage
        )

    def _parse_stream_event(self, event: Any) -> Dict[str, Any]:
        """Parse streaming event"""
        # Return event as-is for now
        # Can be enhanced to normalize different event types
        return {
            "type": event.type,
            "data": event
        }
