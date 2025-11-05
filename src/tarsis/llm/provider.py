"""
LLM Provider abstraction layer.

Provides a unified interface for multiple LLM providers (Anthropic, OpenAI, Ollama, etc.)
using a provider pattern for flexibility and extensibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator
from enum import Enum


class ModelProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


@dataclass
class ModelInfo:
    """Information about a specific model"""
    id: str  # Model identifier (e.g., "claude-sonnet-4-20250514")
    name: str  # Display name
    provider: ModelProvider
    context_window: int  # Maximum context window size
    supports_tools: bool = True  # Whether model supports tool calling
    supports_streaming: bool = True


@dataclass
class Message:
    """Represents a message in the conversation"""
    role: str  # "user", "assistant", "system"
    content: Any  # String or list of content blocks


@dataclass
class Usage:
    """Token usage information"""
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class AssistantMessage:
    """Response from the LLM"""
    content: Any  # String or list of content blocks (including tool uses)
    stop_reason: Optional[str] = None  # Why the model stopped (end_turn, tool_use, etc.)
    usage: Optional[Usage] = None


class ILLMProvider(ABC):
    """
    Base interface for LLM providers.

    All providers must implement this interface to be usable by the agent.
    """

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Get information about the model"""
        pass

    @abstractmethod
    async def create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> AssistantMessage:
        """
        Create a message (request-response).

        Args:
            system_prompt: System prompt for the model
            messages: Conversation history
            tools: Available tools (in Anthropic format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            AssistantMessage with model response
        """
        pass

    @abstractmethod
    async def create_message_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Create a message with streaming response.

        Args:
            Same as create_message

        Yields:
            Stream events (chunks, usage, etc.)
        """
        pass

    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming"""
        return self.model_info.supports_streaming

    def supports_tools(self) -> bool:
        """Whether this provider supports tool calling"""
        return self.model_info.supports_tools


class BaseLLMProvider(ILLMProvider):
    """
    Base implementation with common functionality.

    Subclasses implement provider-specific logic.
    """

    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate provider configuration"""
        if not self.model_id:
            raise ValueError("model_id is required")

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages for the provider's API.
        Default implementation returns as-is.
        """
        return messages

    def _parse_response(self, response: Any) -> AssistantMessage:
        """
        Parse provider response into AssistantMessage.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement _parse_response")
