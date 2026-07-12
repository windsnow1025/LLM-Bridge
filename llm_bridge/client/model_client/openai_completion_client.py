from typing import AsyncGenerator, Literal

import openai.lib.azure
import openai
from openai import Omit
from openai.types.chat import ChatCompletionAudioParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import ResponseFormatJSONSchema

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.openai_completion_message import OpenAICompletionMessage


class OpenAICompletionClient(ChatClient):
    def __init__(
            self,
            model: str,
            messages: list[OpenAICompletionMessage],
            temperature: float | Omit,
            api_type: str,
            client: openai.AsyncOpenAI | openai.lib.azure.AsyncAzureOpenAI,
            reasoning_effort: ReasoningEffort | Omit,
            modalities: list[Literal["text", "audio"]] | Omit,
            audio: ChatCompletionAudioParam | Omit,
            response_format: ResponseFormatJSONSchema | Omit,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.api_type = api_type
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.modalities = modalities
        self.audio = audio
        self.response_format = response_format

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError

    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
