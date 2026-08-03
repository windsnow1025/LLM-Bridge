import logging
from collections.abc import AsyncGenerator, AsyncIterator

from google.genai.types import GenerateContentResponse

from llm_bridge.client.implementations.gemini.gemini_response_handler import GeminiResponseHandler
from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.model_client.gemini_client import GeminiClient
from llm_bridge.type.chat_response import ChatResponse


class StreamGeminiClient(GeminiClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            response: AsyncIterator[GenerateContentResponse] = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=self.messages,
                config=self.config,
            )
        except Exception as e:
            raise_http_exception(e)

        try:
            response_handler = GeminiResponseHandler()
            async for response_delta in response:
                yield await response_handler.process_gemini_response(response_delta)
        except Exception as e:
            logging.exception(e)
            yield ChatResponse(error=repr(e))
