from typing import AsyncGenerator, Iterable, Type, Any
from pydantic import BaseModel

import openai.lib.azure
from openai.types import Reasoning
from openai.types.responses import ToolParam, ResponseIncludable

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
            tools: Iterable[ToolParam],
            reasoning: Reasoning,
            include: list[ResponseIncludable],
            structured_output_base_model: Type[BaseModel] | None = None,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.api_type = api_type
        self.client = client
        self.tools = tools
        self.reasoning = reasoning
        self.include = include
        self.structured_output_base_model = structured_output_base_model

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError

    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
