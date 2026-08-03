import logging
from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types.beta import BetaRawMessageStreamEvent

from llm_bridge.client.implementations.claude.claude_response_handler import ClaudeResponseHandler
from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.model_client.claude_client import ClaudeClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


async def generate_chunk(
        stream: AsyncStream[BetaRawMessageStreamEvent],
        client: AsyncAnthropic,
) -> AsyncGenerator[ChatResponse, None]:
    try:
        response_handler = ClaudeResponseHandler()
        async for event in stream:
            yield await response_handler.process_claude_stream_response(
                event=event,
                client=client,
            )
    except Exception as e:
        logging.exception(e)
        yield ChatResponse(error=repr(e))


class StreamClaudeClient(ClaudeClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            stream: AsyncStream[BetaRawMessageStreamEvent] = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                messages=serialize(self.messages),
                betas=self.betas,
                tools=self.tools,
                cache_control=self.cache_control,
                thinking=self.thinking,
                output_config=self.output_config,
                stream=True,
            )
        except Exception as e:
            raise_http_exception(e)

        async for chunk in generate_chunk(stream, self.client):
            yield chunk
