from typing import AsyncGenerator, Iterable

import openai.lib.azure
from openai.types.responses import ToolParam

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.openai_message import OpenAIMessage


class OpenAIClient(ChatClient):
    def __init__(
            self,
            model: str,
            messages: list[OpenAIMessage],
            temperature: float,
            api_type: str,
            client: openai.AsyncOpenAI | openai.lib.azure.AsyncAzureOpenAI,
            tools: Iterable[ToolParam],
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.api_type = api_type
        self.client = client
        self.tools = tools

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
