import logging

from google.genai.types import GenerateContentResponse

from llm_bridge.client.implementations.gemini.gemini_response_handler import *
from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.model_client.gemini_client import GeminiClient
from llm_bridge.type.chat_response import ChatResponse


class NonStreamGeminiClient(GeminiClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            response: GenerateContentResponse = await self.client.aio.models.generate_content(
                model=self.model,
                contents=self.messages,
                config=self.config,
            )

            gemini_response_handler = GeminiResponseHandler()
            return await gemini_response_handler.process_gemini_response(response)

        except Exception as e:
            raise_http_exception(e)
