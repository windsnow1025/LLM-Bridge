from typing import AsyncGenerator, Iterable

import openai.lib.azure
from openai.types import Reasoning
from openai.types.responses import ToolParam, ResponseIncludable, ResponseTextConfigParam

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.openai_message import OpenAIMessage
from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage


class OpenAIClient(ChatClient):
    def __init__(
            self,
            model: str,
            messages: list[OpenAIMessage | OpenAIResponsesMessage],
            temperature: float,
            api_type: str,
            client: openai.AsyncOpenAI | openai.lib.azure.AsyncAzureOpenAI,
            tools: Iterable[ToolParam] | None = None,
            reasoning: Reasoning | None = None,
            include: list[ResponseIncludable] | None = None,
            text: ResponseTextConfigParam | None = None,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.api_type = api_type
        self.client = client
        self.tools = tools
        self.reasoning = reasoning
        self.include = include
        self.text = text

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError

    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
