from typing import AsyncGenerator

import openai.lib.azure

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.gpt_message import GptMessage


class GPTClient(ChatClient):
    def __init__(
            self,
            model: str,
            messages: list[GptMessage],
            temperature: float,
            api_type: str,
            client: openai.AsyncOpenAI | openai.lib.azure.AsyncAzureOpenAI,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.api_type = api_type
        self.client = client

    async def generate_response(self) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
