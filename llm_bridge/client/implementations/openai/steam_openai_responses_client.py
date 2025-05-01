import logging
import re
from typing import AsyncGenerator

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError, AsyncStream
from openai.types.responses import WebSearchToolParam, ResponseStreamEvent

from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


def process_delta(event: ResponseStreamEvent) -> str:
    if event.type != "response.output_text.delta":
        return ""

    content_delta = event.delta
    return content_delta


async def generate_chunk(
        stream: AsyncStream[ResponseStreamEvent]
) -> AsyncGenerator[ChatResponse, None]:
    try:
        async for event in stream:
            content_delta = process_delta(event)
            yield ChatResponse(text=content_delta)
    except Exception as e:
        logging.exception(e)
        yield ChatResponse(error=repr(e))


class StreamOpenAIResponsesClient(OpenAIClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")
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

        async for chunk in generate_chunk(stream):
            yield chunk