import logging
from collections.abc import AsyncGenerator

from openai import AsyncStream
from openai.types.responses import ResponseStreamEvent

from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.implementations.openai_responses.openai_responses_response_handler import \
    process_openai_responses_stream_response
from llm_bridge.client.model_client.openai_responses_client import OpenAIResponsesClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


async def generate_chunk(
        stream: AsyncStream[ResponseStreamEvent],
) -> AsyncGenerator[ChatResponse, None]:
    try:
        async for event in stream:
            yield await process_openai_responses_stream_response(event)
    except Exception as e:
        logging.exception(e)
        yield ChatResponse(error=repr(e))


class StreamOpenAIResponsesClient(OpenAIResponsesClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            stream: AsyncStream[ResponseStreamEvent] = await self.client.responses.create(
                model=self.model,
                input=serialize(self.messages),
                temperature=self.temperature,
                stream=True,
                tools=self.tools,
                reasoning=self.reasoning,
                include=self.include,
                text=self.text,
            )

        except Exception as e:
            raise_http_exception(e)

        async for chunk in generate_chunk(stream):
            yield chunk
