import logging
import re
from pprint import pprint
from typing import AsyncGenerator, Optional

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError, AsyncStream
from openai.types.responses import ResponseStreamEvent

from llm_bridge.client.implementations.openai.openai_token_couter import count_openai_responses_input_tokens, \
    count_openai_output_tokens
from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse, Citation, File
from llm_bridge.type.serializer import serialize


def process_delta(event: ResponseStreamEvent) -> ChatResponse:
    text: str = ""
    files: list[File] = []
    citations: list[Citation] = []

    if event.type == "response.output_text.delta":
        text = event.delta
    # Citation is unavailable in OpenAI Responses API
    if event.type == "response.output_text.annotation.added":
        pass
    # Image Generation untestable due to organization verification requirement
    # if event.type == "response.image_generation_call.partial_image":
    #     file = File(
    #         name="generated_image.png",
    #         data=event.partial_image_b64,
    #         type="image/png",
    #     )
    #     files.append(file)

    chat_response = ChatResponse(
        text=text,
        files=files,
        citations=citations,
    )
    return chat_response


async def generate_chunk(
        stream: AsyncStream[ResponseStreamEvent],
        input_tokens: int,
) -> AsyncGenerator[ChatResponse, None]:
    try:
        async for event in stream:
            chat_response = process_delta(event)
            output_tokens = count_openai_output_tokens(chat_response)
            yield ChatResponse(
                text=chat_response.text,
                files=chat_response.files,
                citations=chat_response.citations,
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
                reasoning=self.reasoning,
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