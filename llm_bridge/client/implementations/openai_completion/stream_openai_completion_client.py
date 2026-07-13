import logging
import re
from collections.abc import AsyncGenerator

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from llm_bridge.client.model_client.openai_completion_client import OpenAICompletionClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


def process_delta(completion_delta: ChatCompletionChunk) -> tuple[str, str]:
    # Necessary for Azure
    if not completion_delta.choices:
        return "", ""

    delta: ChoiceDelta = completion_delta.choices[0].delta

    content_delta = delta.content
    if not content_delta:
        content_delta = ""

    # Audio field untyped in the SDK
    audio_delta = ""
    audio: dict | None = getattr(delta, "audio", None)
    if audio is not None:
        content_delta += audio.get("transcript") or ""
        audio_delta = audio.get("data") or ""

    logging.debug(f"chunk: {content_delta}")
    return content_delta, audio_delta


async def generate_chunk(
        completion: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[ChatResponse, None]:
    try:
        async for completion_delta in completion:
            content_delta, audio_delta = process_delta(completion_delta)
            input_tokens = 0
            output_tokens = 0
            if completion_delta.usage is not None:
                input_tokens = completion_delta.usage.prompt_tokens
                output_tokens = completion_delta.usage.completion_tokens
            yield ChatResponse(
                text=content_delta,
                audio=audio_delta or None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
    except Exception as e:
        logging.exception(e)
        yield ChatResponse(error=repr(e))


class StreamOpenAICompletionClient(OpenAICompletionClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            completion: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
                messages=serialize(self.messages),
                model=self.model,
                temperature=self.temperature,
                stream=True,
                stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
                reasoning_effort=self.reasoning_effort,
                modalities=self.modalities,
                audio=self.audio,
                response_format=self.response_format,
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

        async for chunk in generate_chunk(completion):
            yield chunk
