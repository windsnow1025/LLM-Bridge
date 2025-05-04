import logging
import re
from typing import AsyncGenerator

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError, AsyncStream
from openai.types.responses import ResponseStreamEvent

from llm_bridge.client.implementations.openai.openai_token_couter import count_openai_responses_input_tokens, \
    count_openai_output_tokens
from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


def process_delta(event: ResponseStreamEvent) -> str:
    if event.type != "response.output_text.delta":
        return ""

    content_delta = event.delta
    return content_delta


async def generate_chunk(
        stream: AsyncStream[ResponseStreamEvent],
        input_tokens: int,
) -> AsyncGenerator[ChatResponse, None]:
    try:
        async for event in stream:
            content_delta = process_delta(event)
            chat_response = ChatResponse(text=content_delta)
            output_tokens = count_openai_output_tokens(chat_response)
            yield ChatResponse(
                text=content_delta,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
    except Exception as e:
        logging.exception(e)
        yield ChatResponse(error=repr(e))


class StreamOpenAIResponsesClient(OpenAIClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            input_tokens = count_openai_responses_input_tokens(
                messages=self.messages
            )

            stream: AsyncStream[ResponseStreamEvent] = await self.client.responses.create(
                model=self.model,
                input=serialize(self.messages),
                temperature=self.temperature,
                stream=True,
                tools=self.tools,
            )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            text = e.response.text
            raise HTTPException(status_code=status_code, detail=text)
        except openai.BadRequestError as e:
            status_code = e.status_code
            text = e.message
            raise HTTPException(status_code=status_code, detail=text)
        except APIStatusError as e:
            status_code = e.status_code
            text = e.message
            raise HTTPException(status_code=status_code, detail=text)
        except Exception as e:
            logging.exception(e)
            match = re.search(r'\d{3}', str(e))
            if match:
                error_code = int(match.group(0))
            else:
                error_code = 500

            raise HTTPException(status_code=error_code, detail=str(e))

        async for chunk in generate_chunk(stream, input_tokens):
            yield chunk