from typing import AsyncGenerator, Any

import anthropic
from anthropic.types import ThinkingConfigEnabledParam, AnthropicBetaParam
from anthropic.types.beta import BetaToolUnionParam

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.claude_message import ClaudeMessage


class ClaudeClient(ChatClient):
    def __init__(
            self,
            model: str,
            messages: list[ClaudeMessage],
            temperature: float,
            system: str,
            client: anthropic.AsyncAnthropic,
            max_tokens: int,
            thinking: ThinkingConfigEnabledParam,
            betas: list[AnthropicBetaParam],
            input_tokens: int,
            tools: list[BetaToolUnionParam],
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.system = system
        self.client = client
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.betas = betas
        self.input_tokens = input_tokens
        self.tools = tools

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
