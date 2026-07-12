import re
from typing import Any, Literal

import openai
from fastapi import HTTPException
from openai import Omit
from openai.types.chat import ChatCompletionAudioParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from llm_bridge.client.implementations.openai_completion.non_stream_openai_completion_client import \
    NonStreamOpenAICompletionClient
from llm_bridge.client.implementations.openai_completion.stream_openai_completion_client import \
    StreamOpenAICompletionClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_openai
from llm_bridge.type.message import Message


async def create_openai_completion_client(
        api_keys: dict[str, str],
        messages: list[Message],
        model: str,
        api_type: str,
        temperature: float,
        stream: bool,
        thought: bool,
        structured_output_schema: dict[str, Any] | None,
) -> StreamOpenAICompletionClient | NonStreamOpenAICompletionClient:
    omit = Omit()

    if api_type == "OpenAI":
        client = openai.AsyncOpenAI(
            api_key=api_keys["OPENAI_API_KEY"],
        )
    elif api_type == "OpenAI-GitHub":
        client = openai.AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=api_keys["GITHUB_API_KEY"],
        )
    else:
        raise HTTPException(status_code=500, detail="API Type not matched")

    openai_messages = await convert_messages_to_openai(messages)

    reasoning_effort: ReasoningEffort | Omit = omit
    if re.match(r"gpt-5.*", model):
        if thought:
            reasoning_effort = "high"

    modalities: list[Literal["text", "audio"]] | Omit = omit
    audio: ChatCompletionAudioParam | Omit = omit
    temperature_param: float | Omit = temperature
    if re.match(r"gpt-audio.*", model):
        modalities = ["text", "audio"]
        audio = ChatCompletionAudioParam(
            voice="marin",
            format="pcm16" if stream else "wav", # streaming requires a raw format
        )
        temperature_param = omit # temperature 0 degenerates audio generation into an endless stream

    response_format: ResponseFormatJSONSchema | Omit = omit
    if structured_output_schema:
        response_format = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="structured_output",
                schema=structured_output_schema,
                strict=True,
            ),
        )

    if stream:
        return StreamOpenAICompletionClient(
            model=model,
            messages=openai_messages,
            temperature=temperature_param,
            api_type=api_type,
            client=client,
            reasoning_effort=reasoning_effort,
            modalities=modalities,
            audio=audio,
            response_format=response_format,
        )
    else:
        return NonStreamOpenAICompletionClient(
            model=model,
            messages=openai_messages,
            temperature=temperature_param,
            api_type=api_type,
            client=client,
            reasoning_effort=reasoning_effort,
            modalities=modalities,
            audio=audio,
            response_format=response_format,
        )
