"""
LLM module - Provides LLM provider abstractions.
"""

from .provider import (
    ILLMProvider,
    BaseLLMProvider,
    ModelInfo,
    ModelProvider,
    Message,
    Usage,
    AssistantMessage
)
from .anthropic_provider import AnthropicProvider
from .ollama_provider import OllamaProvider
from .gemini_provider import GeminiProvider


def create_llm_provider(
    provider_type: str = "ollama",
    model_id: str = None,
    api_key: str = None,
    **kwargs
) -> ILLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider_type: Type of provider ("anthropic", "ollama", "gemini")
        model_id: Model identifier
        api_key: API key (if required)
        **kwargs: Additional provider-specific arguments

    Returns:
        ILLMProvider instance

    Raises:
        ValueError: If provider_type is unknown
    """
    provider_type = provider_type.lower()

    if provider_type == "anthropic":
        model_id = model_id or "claude-3-5-sonnet-20241022"
        return AnthropicProvider(model_id=model_id, api_key=api_key)

    elif provider_type == "ollama":
        model_id = model_id or "qwen2.5-coder:7b"
        return OllamaProvider(
            model_id=model_id,
            base_url=kwargs.get("base_url"),
            api_key=api_key
        )

    elif provider_type == "gemini":
        model_id = model_id or "gemini-2.5-flash"
        return GeminiProvider(model_id=model_id, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported: anthropic, ollama, gemini"
        )


__all__ = [
    # Base classes
    "ILLMProvider",
    "BaseLLMProvider",
    "ModelInfo",
    "ModelProvider",
    "Message",
    "Usage",
    "AssistantMessage",
    # Providers
    "AnthropicProvider",
    "OllamaProvider",
    "GeminiProvider",
    # Factory
    "create_llm_provider",
]
