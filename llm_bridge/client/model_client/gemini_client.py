from typing import AsyncGenerator

from google import genai
from google.genai import types

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.gemini_message import GeminiMessage


class GeminiClient(ChatClient):
    def __init__(
            self,
            model: str,
            messages: list[GeminiMessage],
            temperature: float,
            client: genai.Client,
            config: types.GenerateContentConfig,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.client = client
        self.config = config

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
