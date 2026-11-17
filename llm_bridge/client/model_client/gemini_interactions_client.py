from collections.abc import AsyncGenerator

from google import genai
from google.genai.interactions import ToolParam, GenerationConfigParam, SafetySettingParam, ServiceTier, \
    TextResponseFormatParam

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.gemini_interactions_message import GeminiInteractionsMessage


class GeminiInteractionsClient(ChatClient):
    def __init__(
            self,
            model: str,
            system_instruction: str,
            messages: list[GeminiInteractionsMessage],
            client: genai.Client,
            tools: list[ToolParam],
            generation_config: GenerationConfigParam,
            safety_settings: list[SafetySettingParam],
            service_tier: ServiceTier,
            response_format: TextResponseFormatParam,
    ):
        self.model = model
        self.system_instruction = system_instruction
        self.messages = messages
        self.client = client
        self.tools = tools
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.service_tier = service_tier
        self.response_format = response_format

    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
