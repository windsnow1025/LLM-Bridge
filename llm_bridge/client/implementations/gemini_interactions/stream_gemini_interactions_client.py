import logging
from collections.abc import AsyncGenerator, AsyncIterator

from google.genai.interactions import InteractionSSEEvent

from llm_bridge.client.implementations.gemini_interactions.gemini_interactions_response_handler import \
    GeminiInteractionsStreamResponseHandler
from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.model_client.gemini_interactions_client import GeminiInteractionsClient
from llm_bridge.type.chat_response import ChatResponse


class StreamGeminiInteractionsClient(GeminiInteractionsClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            stream: AsyncIterator[InteractionSSEEvent] = await self.client.aio.interactions.create(
                model=self.model,
                store=False,
                system_instruction=self.system_instruction,
                input=self.messages,
                stream=True,
                tools=self.tools,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                service_tier=self.service_tier,
                response_format=self.response_format,
            )
        except Exception as e:
            raise_http_exception(e)

        try:
            response_handler = GeminiInteractionsStreamResponseHandler()
            async for event in stream:
                yield await response_handler.process_gemini_interactions_stream_response(event)
        except Exception as e:
            logging.exception(e)
            yield ChatResponse(error=repr(e))
