from typing import AsyncGenerator

import anthropic
from anthropic import Omit
from anthropic.types import AnthropicBetaParam
from anthropic.types.beta import BetaToolUnionParam, BetaOutputConfigParam, BetaThinkingConfigParam, \
    BetaCacheControlEphemeralParam

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
            betas: list[AnthropicBetaParam],
            tools: list[BetaToolUnionParam],
            cache_control: BetaCacheControlEphemeralParam,
            thinking: BetaThinkingConfigParam | Omit,
            output_config: BetaOutputConfigParam | Omit,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.system = system
        self.client = client
        self.max_tokens = max_tokens
        self.betas = betas
        self.tools = tools
        self.cache_control = cache_control
        self.thinking = thinking
        self.output_config = output_config

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError

    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
