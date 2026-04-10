from typing import AsyncGenerator

import xai_sdk
from xai_sdk.chat import ReasoningEffort

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.xai_message import XAIMessage


class XAIClient(ChatClient):
    def __init__(
            self,
            model: str,
            messages: list[XAIMessage],
            temperature: float,
            client: xai_sdk.AsyncClient,
            reasoning_effort: ReasoningEffort | None,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.client = client
        self.reasoning_effort = reasoning_effort

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError

    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
