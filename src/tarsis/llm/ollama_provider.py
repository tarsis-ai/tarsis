"""
Ollama LLM Provider implementation for local models.
"""

import os
import httpx
import json
import re
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

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


class OllamaProvider(BaseLLMProvider):
    """
    Provider for Ollama local models.

    Supports models like:
    - Llama 3.1 / 3.2
    - Qwen 2.5 Coder
    - DeepSeek Coder
    - CodeLlama
    - And any other Ollama-compatible model

    Note: Tool calling support varies by model
    """

    def __init__(
        self,
        model_id: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        use_structured_output: Optional[bool] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize Ollama provider.

        Args:
            model_id: Ollama model name (e.g., "llama3.1:8b", "qwen2.5-coder:7b")
            base_url: Ollama server URL
            api_key: Optional API key (not usually needed for local Ollama)
            use_structured_output: Whether to use structured output (grammar-based tool calling).
                                   If None, defaults based on OLLAMA_STRUCTURED_OUTPUT env var (default: False).
                                   Set to False to avoid llama.cpp grammar crashes with complex schemas.
            timeout: Request timeout in seconds. If None, reads from OLLAMA_TIMEOUT env var (default: 1800).
                    Set to 0 for unlimited timeout (only recommended for local development).
        """
        super().__init__(model_id, api_key)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Configure timeout
        if timeout is None:
            # Read from environment variable, default to 1800 seconds (30 minutes)
            timeout_str = os.getenv("OLLAMA_TIMEOUT", "1800")
            try:
                timeout = float(timeout_str)
            except ValueError:
                logger.warning(f"Invalid OLLAMA_TIMEOUT value: {timeout_str}, using default 1800")
                timeout = 1800.0

        # Create granular timeout configuration
        # - connect: 10s is reasonable for local connection
        # - read: configurable (main timeout), 0 means unlimited
        # - write: 30s should be enough for any request
        # - pool: 10s for acquiring connection from pool
        if timeout == 0:
            # Unlimited read timeout for local CPU-based inference
            timeout_config = httpx.Timeout(
                connect=10.0,
                read=None,  # No read timeout
                write=30.0,
                pool=10.0
            )
            logger.info("Ollama provider initialized with unlimited read timeout (CPU inference mode)")
        else:
            timeout_config = httpx.Timeout(
                connect=10.0,
                read=timeout,
                write=30.0,
                pool=10.0
            )
            logger.info(f"Ollama provider initialized with {timeout}s read timeout")

        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_config)

        # Determine if we should use structured output (grammar-based tool calling)
        # Default to False to avoid grammar parser crashes
        if use_structured_output is None:
            env_value = os.getenv("OLLAMA_STRUCTURED_OUTPUT", "false").lower()
            self.use_structured_output = env_value in ("true", "1", "yes")
        else:
            self.use_structured_output = use_structured_output

    @property
    def model_info(self) -> ModelInfo:
        """Get model information"""
        # Parse model name to determine capabilities
        # Format is usually "model:tag" like "qwen2.5-coder:7b"
        supports_tools = self._check_tool_support(self.model_id)

        return ModelInfo(
            id=self.model_id,
            name=self.model_id,
            provider=ModelProvider.OLLAMA,
            context_window=self._estimate_context_window(),
            supports_tools=supports_tools,
            supports_streaming=True
        )

    def _check_tool_support(self, model_id: str) -> bool:
        """
        Check if model supports tool calling.

        Models known to support tools:
        - qwen2.5-coder
        - llama3.1 (8B and larger)
        - mistral (some versions)
        """
        model_lower = model_id.lower()
        tool_capable_models = ["qwen2.5", "llama3.1", "llama3.2"]
        return any(name in model_lower for name in tool_capable_models)

    def _estimate_context_window(self) -> int:
        """Estimate context window based on model name"""
        model_lower = self.model_id.lower()

        # Common context windows
        if "32k" in model_lower or "32768" in model_lower:
            return 32768
        elif "16k" in model_lower:
            return 16384
        elif "8k" in model_lower:
            return 8192

        # Defaults for common models
        if "qwen" in model_lower:
            return 32768
        elif "llama3" in model_lower:
            return 8192

        # Conservative default
        return 4096

    def _tools_to_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Convert tools to a text prompt for prompt-based tool calling.

        This is used when use_structured_output=False to avoid grammar crashes.
        The model is instructed to call tools by outputting JSON in a specific format.
        """
        if not tools:
            return ""

        prompt = "\n\n" + "="*80 + "\n"
        prompt += "TOOL CALLING INSTRUCTIONS\n"
        prompt += "="*80 + "\n\n"

        prompt += "You have access to tools that you can call to complete tasks.\n\n"
        prompt += "**HOW TO CALL A TOOL:**\n\n"
        prompt += "Output a JSON object in this EXACT format (you can wrap it in markdown code blocks):\n\n"
        prompt += "```json\n"
        prompt += '{"tool": "tool_name", "input": {"param1": "value1"}}\n'
        prompt += "```\n\n"

        prompt += "**IMPORTANT RULES:**\n"
        prompt += "1. Call ONE tool at a time\n"
        prompt += "2. Wait for the tool result before calling another tool\n"
        prompt += "3. Use the 'attempt_completion' tool when you finish the task\n"
        prompt += "4. Output ONLY the JSON - no explanations before or after\n\n"

        prompt += "="*80 + "\n"
        prompt += "AVAILABLE TOOLS\n"
        prompt += "="*80 + "\n\n"

        for i, tool in enumerate(tools, 1):
            prompt += f"## {i}. {tool['name']}\n\n"
            prompt += f"{tool['description']}\n\n"

            schema = tool.get('input_schema', {})
            properties = schema.get('properties', {})
            required = schema.get('required', [])

            if properties:
                prompt += "**Parameters:**\n\n"
                for param_name, param_info in properties.items():
                    is_required = param_name in required
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')

                    marker = "REQUIRED" if is_required else "optional"
                    prompt += f"- **{param_name}** ({param_type}) [{marker}]\n"
                    prompt += f"  {param_desc}\n"

                    # Add enum values if present
                    if 'enum' in param_info:
                        prompt += f"  Allowed values: {', '.join(map(str, param_info['enum']))}\n"

                    prompt += "\n"
            else:
                prompt += "**No parameters required**\n\n"

            # Add example if this is attempt_completion (most important)
            if tool['name'] == 'attempt_completion':
                prompt += "**Example:**\n"
                prompt += "```json\n"
                prompt += '{"tool": "attempt_completion", "input": {"result": "Created hello.py with hello world program"}}\n'
                prompt += "```\n\n"

            prompt += "-" * 80 + "\n\n"

        return prompt

    def _parse_text_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from text output (prompt-based tool calling).

        Looks for JSON blocks in the format:
        {"tool": "tool_name", "input": {...}}

        Returns:
            List of tool call dictionaries with 'name' and 'input' fields
        """
        tool_calls = []

        # Strategy 1: Try to find JSON blocks in markdown code blocks
        json_block_pattern = r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```'
        matches = re.finditer(json_block_pattern, text, re.DOTALL)

        for match in matches:
            try:
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)

                if isinstance(parsed, dict) and 'tool' in parsed:
                    tool_calls.append({
                        'name': parsed['tool'],
                        'input': parsed.get('input', {})
                    })
                    logger.debug(f"Parsed tool call from markdown block: {parsed['tool']}")
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")
                continue

        # Strategy 2: If no code blocks found, try to extract JSON with balanced braces
        if not tool_calls:
            # Find all potential JSON objects
            brace_depth = 0
            json_start = -1

            for i, char in enumerate(text):
                if char == '{':
                    if brace_depth == 0:
                        json_start = i
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and json_start >= 0:
                        # Found a complete JSON object
                        json_str = text[json_start:i+1]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and 'tool' in parsed:
                                tool_calls.append({
                                    'name': parsed['tool'],
                                    'input': parsed.get('input', {})
                                })
                                logger.debug(f"Parsed tool call from raw JSON: {parsed['tool']}")
                        except json.JSONDecodeError:
                            pass
                        json_start = -1

        # Strategy 3: Log the text if no tool calls found for debugging
        if not tool_calls:
            logger.debug(f"No tool calls found in text. First 300 chars: {text[:300]}")

        return tool_calls

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
        Create a message using Ollama API.

        Uses Ollama's /api/chat endpoint.
        Supports both structured output (grammar-based) and prompt-based tool calling.
        """
        # Determine if we should use structured output or prompt-based approach
        use_structured = self.use_structured_output and tools and self.model_info.supports_tools

        # Modify system prompt if using prompt-based tool calling
        effective_system_prompt = system_prompt
        if tools and not use_structured:
            logger.info("Using prompt-based tool calling (structured output disabled)")
            effective_system_prompt = system_prompt + self._tools_to_prompt(tools)

        # Format messages for Ollama
        ollama_messages = self._format_messages_for_ollama(effective_system_prompt, messages)

        # Build request
        request_data = {
            "model": self.model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # Add tools if using structured output
        if use_structured:
            request_data["tools"] = self._convert_tools_to_ollama_format(tools)
            logger.info("Using structured output (grammar-based tool calling)")

        # Make API call with error handling and fallback
        try:
            logger.debug(f"Sending request to Ollama: {len(ollama_messages)} messages, "
                        f"use_structured={use_structured}")

            response = await self.client.post("/api/chat", json=request_data)
            response.raise_for_status()

            response_data = response.json()

            # Log response metadata
            if "prompt_eval_count" in response_data:
                logger.debug(f"Token usage: prompt={response_data.get('prompt_eval_count')}, "
                           f"completion={response_data.get('eval_count')}")

            # Parse response
            return self._parse_ollama_response(response_data, use_prompt_based=not use_structured)

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a grammar-related error
            if use_structured and ("grammar" in error_msg or "unexpected empty" in error_msg):
                logger.warning(
                    f"Grammar parser error detected, falling back to prompt-based tool calling: {e}"
                )

                # Retry with prompt-based approach
                effective_system_prompt = system_prompt + self._tools_to_prompt(tools)
                ollama_messages = self._format_messages_for_ollama(effective_system_prompt, messages)

                # Remove tools from request
                request_data["messages"] = ollama_messages
                if "tools" in request_data:
                    del request_data["tools"]

                # Retry
                response = await self.client.post("/api/chat", json=request_data)
                response.raise_for_status()

                return self._parse_ollama_response(response.json(), use_prompt_based=True)

            # Re-raise if it's not a grammar error
            raise

    async def create_message_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Create a streaming message using Ollama.

        Note: Streaming with prompt-based tool calling is supported but
        tool calls will only be detected after the full message is received.
        """
        # Determine if we should use structured output or prompt-based approach
        use_structured = self.use_structured_output and tools and self.model_info.supports_tools

        # Modify system prompt if using prompt-based tool calling
        effective_system_prompt = system_prompt
        if tools and not use_structured:
            logger.info("Using prompt-based tool calling for streaming (structured output disabled)")
            effective_system_prompt = system_prompt + self._tools_to_prompt(tools)

        # Format messages
        ollama_messages = self._format_messages_for_ollama(effective_system_prompt, messages)

        # Build request
        request_data = {
            "model": self.model_id,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # Add tools if using structured output
        if use_structured:
            request_data["tools"] = self._convert_tools_to_ollama_format(tools)
            logger.info("Using structured output for streaming (grammar-based tool calling)")

        # Stream response with error handling
        try:
            async with self.client.stream("POST", "/api/chat", json=request_data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        chunk = json.loads(line)
                        yield self._parse_stream_chunk(chunk)

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a grammar-related error
            if use_structured and ("grammar" in error_msg or "unexpected empty" in error_msg):
                logger.warning(
                    f"Grammar parser error in streaming, falling back to prompt-based: {e}"
                )

                # Retry with prompt-based approach
                effective_system_prompt = system_prompt + self._tools_to_prompt(tools)
                ollama_messages = self._format_messages_for_ollama(effective_system_prompt, messages)

                # Remove tools from request
                request_data["messages"] = ollama_messages
                if "tools" in request_data:
                    del request_data["tools"]

                # Retry streaming
                async with self.client.stream("POST", "/api/chat", json=request_data) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            chunk = json.loads(line)
                            yield self._parse_stream_chunk(chunk)
            else:
                # Re-raise if it's not a grammar error
                raise

    def _format_messages_for_ollama(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert Anthropic-style messages to Ollama format.

        Ollama uses a simpler format with role and content.
        """
        formatted = []

        # Add system message first
        if system_prompt:
            formatted.append({
                "role": "system",
                "content": system_prompt
            })

        # Convert messages
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Simplify content to string if it's complex
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            # Format tool results as text
                            text_parts.append(f"Tool result: {block.get('content', '')}")
                content = "\n".join(text_parts)

            formatted.append({
                "role": role,
                "content": content
            })

        return formatted

    def _convert_tools_to_ollama_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Anthropic tool format to Ollama/OpenAI format.

        Ollama expects OpenAI-compatible tool format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... }
            }
        }

        Anthropic format is:
        {
            "name": "...",
            "description": "...",
            "input_schema": { ... }
        }
        """
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        return ollama_tools

    def _parse_ollama_response(
        self,
        response: Dict[str, Any],
        use_prompt_based: bool = False
    ) -> AssistantMessage:
        """
        Parse Ollama response into AssistantMessage.

        Args:
            response: Raw Ollama API response
            use_prompt_based: If True, parse tool calls from text content

        Returns:
            AssistantMessage with properly formatted content
        """
        # Debug: Log raw response structure
        logger.debug(f"Raw Ollama response keys: {list(response.keys())}")

        # Ollama now returns OpenAI-compatible format
        # New format: {"choices": [{"message": {...}}]}
        # Old format: {"message": {...}}
        if "choices" in response:
            # OpenAI-compatible format (newer Ollama versions)
            message_data = response["choices"][0]["message"]
            logger.debug("Using OpenAI-compatible response format")
        else:
            # Legacy Ollama format
            message_data = response.get("message", {})
            logger.debug("Using legacy Ollama response format")

        content = message_data.get("content") or ""

        logger.debug(f"Extracted content length: {len(content)} chars")
        if not content:
            logger.debug("Content is empty (may have tool calls instead)")

        # Check for structured tool calls (from grammar-based approach)
        tool_calls = message_data.get("tool_calls", [])
        if tool_calls:
            # Convert to Anthropic-style content blocks
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})

            for tool_call in tool_calls:
                # Parse arguments - in OpenAI format it's a JSON string
                arguments = tool_call.get("function", {}).get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                        logger.debug(f"Parsed tool arguments from JSON string: {arguments}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments JSON: {arguments}, error: {e}")
                        arguments = {}

                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": tool_call.get("function", {}).get("name", ""),
                    "input": arguments
                })
            content = content_blocks
            logger.debug(f"Extracted {len(tool_calls)} structured tool calls")

        # Check for prompt-based tool calls (from text parsing)
        elif use_prompt_based and content:
            parsed_tool_calls = self._parse_text_tool_calls(content)

            if parsed_tool_calls:
                # Convert to Anthropic-style content blocks
                content_blocks = []

                for i, tool_call in enumerate(parsed_tool_calls):
                    content_blocks.append({
                        "type": "tool_use",
                        "id": f"tool_{i}",  # Generate a simple ID
                        "name": tool_call["name"],
                        "input": tool_call["input"]
                    })

                content = content_blocks
            else:
                # No tool calls found, return as text
                content = [{"type": "text", "text": content}]

        elif content:
            # Plain text response
            content = [{"type": "text", "text": content}]
        else:
            # Empty content - return empty list to avoid downstream errors
            logger.error("Ollama returned completely empty content - no text and no tool calls!")
            content = []

        # Extract usage if available
        usage = None
        if "usage" in response:
            # OpenAI-compatible format
            usage_data = response["usage"]
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        elif "prompt_eval_count" in response or "eval_count" in response:
            # Legacy Ollama format
            usage = Usage(
                input_tokens=response.get("prompt_eval_count", 0),
                output_tokens=response.get("eval_count", 0),
                total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            )

        # Extract stop reason
        if "choices" in response:
            # OpenAI-compatible format
            stop_reason = response["choices"][0].get("finish_reason")
        else:
            # Legacy Ollama format
            stop_reason = response.get("done_reason")

        return AssistantMessage(
            content=content,
            stop_reason=stop_reason,
            usage=usage
        )

    def _parse_stream_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a streaming chunk"""
        return {
            "type": "chunk",
            "data": chunk
        }

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
