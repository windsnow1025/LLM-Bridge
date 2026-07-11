import json
from typing import Any

import xai_sdk
from xai_sdk.proto import chat_pb2
from xai_sdk.tools import web_search as web_search_tool, x_search as x_search_tool, \
    code_execution as code_execution_tool
from xai_sdk.types import ReasoningEffort

from llm_bridge.client.implementations.xai.non_stream_xai_client import NonStreamXAIClient
from llm_bridge.client.implementations.xai.stream_xai_client import StreamXAIClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_xai
from llm_bridge.type.message import Message


async def create_xai_client(
        api_key: str,
        messages: list[Message],
        model: str,
        temperature: float,
        stream: bool,
        thought: bool,
        web_search: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None,
) -> StreamXAIClient | NonStreamXAIClient:
    client = xai_sdk.AsyncClient(
        api_key=api_key,
    )

    xai_messages = await convert_messages_to_xai(messages)

    # Reasoning cannot be disabled in xAI
    reasoning_effort: ReasoningEffort = "high"

    tools: list[chat_pb2.Tool] = []
    if web_search:
        tools.append(web_search_tool(
            enable_image_understanding=True
        ))
        tools.append(x_search_tool(
            enable_image_understanding=True,
            enable_video_understanding=True,
        ))
    if code_execution:
        tools.append(code_execution_tool())

    response_format: chat_pb2.ResponseFormat | None = None
    if structured_output_schema:
        response_format = chat_pb2.ResponseFormat(
            format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
            schema=json.dumps(structured_output_schema),
        )

    if stream:
        return StreamXAIClient(
            model=model,
            messages=xai_messages,
            temperature=temperature,
            client=client,
            tools=tools,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )
    else:
        return NonStreamXAIClient(
            model=model,
            messages=xai_messages,
            temperature=temperature,
            client=client,
            tools=tools,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )
