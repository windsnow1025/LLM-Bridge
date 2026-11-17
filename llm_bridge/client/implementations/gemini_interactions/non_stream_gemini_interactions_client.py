import logging

from google.genai.interactions import Interaction

from llm_bridge.client.implementations.gemini_interactions.gemini_interactions_response_handler import \
    process_gemini_interactions_non_stream_response
from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.model_client.gemini_interactions_client import GeminiInteractionsClient
from llm_bridge.type.chat_response import ChatResponse


class NonStreamGeminiInteractionsClient(GeminiInteractionsClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            interaction: Interaction = await self.client.aio.interactions.create(
                model=self.model,
                store=False,
                system_instruction=self.system_instruction,
                input=self.messages,
                tools=self.tools,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                service_tier=self.service_tier,
                response_format=self.response_format,
            )

            return await process_gemini_interactions_non_stream_response(
                interaction=interaction,
            )

        except Exception as e:
            raise_http_exception(e)
