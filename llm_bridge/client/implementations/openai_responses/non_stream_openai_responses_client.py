import logging

from openai.types.responses import Response

from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.implementations.openai_responses.openai_responses_response_handler import \
    process_openai_responses_non_stream_response
from llm_bridge.client.model_client.openai_responses_client import OpenAIResponsesClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


class NonStreamOpenAIResponsesClient(OpenAIResponsesClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            response: Response = await self.client.responses.create(
                model=self.model,
                input=serialize(self.messages),
                temperature=self.temperature,
                stream=False,
                tools=self.tools,
                reasoning=self.reasoning,
                include=self.include,
                text=self.text,
            )

            return await process_openai_responses_non_stream_response(
                response=response,
            )
        except Exception as e:
            raise_http_exception(e)
