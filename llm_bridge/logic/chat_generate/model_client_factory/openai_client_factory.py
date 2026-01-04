import re
from typing import Any

import openai
from fastapi import HTTPException
from openai import Omit
from openai.types import Reasoning
from openai.types.responses import WebSearchToolParam, ResponseIncludable
from openai.types.responses.tool_param import CodeInterpreter, CodeInterpreterContainerCodeInterpreterToolAuto, \
    ImageGeneration, ToolParam

from llm_bridge.client.implementations.openai.non_stream_openai_client import NonStreamOpenAIClient
from llm_bridge.client.implementations.openai.non_stream_openai_responses_client import NonStreamOpenAIResponsesClient
from llm_bridge.client.implementations.openai.steam_openai_responses_client import StreamOpenAIResponsesClient
from llm_bridge.client.implementations.openai.stream_openai_client import StreamOpenAIClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_openai_responses, \
    convert_messages_to_openai
from llm_bridge.logic.chat_generate.model_client_factory.schema_converter import json_schema_to_pydantic_model
from llm_bridge.type.message import Message


async def create_openai_client(
        api_keys: dict,
        messages: list[Message],
        model: str,
        api_type: str,
        temperature: float,
        stream: bool,
        thought: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None,
):
    omit = Omit()

    if api_type == "OpenAI":
        client = openai.AsyncOpenAI(
            api_key=api_keys["OPENAI_API_KEY"],
        )
    elif api_type == "OpenAI-Azure":
        client = openai.AsyncAzureOpenAI(
            api_version="2025-03-01-preview",
            azure_endpoint=api_keys["AZURE_API_BASE"],
            api_key=api_keys["AZURE_API_KEY"],
        )
    elif api_type == "OpenAI-GitHub":
        client = openai.AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=api_keys["GITHUB_API_KEY"],
        )
    elif api_type == "Grok":
        client = openai.AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_keys["XAI_API_KEY"],
        )
    else:
        raise HTTPException(status_code=500, detail="API Type not matched")

    if api_type in ("OpenAI", "OpenAI-Azure"):
        use_responses_api = True
    else:
        use_responses_api = False

    if use_responses_api:
        openai_messages = await convert_messages_to_openai_responses(messages)
    else:
        openai_messages = await convert_messages_to_openai(messages)

    tools: list[ToolParam] = []
    reasoning: Reasoning | Omit = omit
    include: list[ResponseIncludable] = ["code_interpreter_call.outputs"]

    if model not in ["gpt-5-pro", "gpt-5.2-pro"] and "codex" not in model:
        if code_execution:
            tools.append(
                CodeInterpreter(
                    type="code_interpreter",
                    container=CodeInterpreterContainerCodeInterpreterToolAuto(type="auto")
                )
            )
    tools.append(
        WebSearchToolParam(
            type="web_search",
            search_context_size="high",
        )
    )
    if re.match(r"gpt-5.*", model):
        temperature = 1
    if re.match(r"gpt-5.*", model):
        if thought:
            reasoning = Reasoning(
                effort="high",
                summary="auto",
            )
    if re.match(r"gpt-5.*", model) and "codex" not in model:
        tools.append(
            ImageGeneration(
                type="image_generation",
            )
        )

    structured_output_base_model = None
    if structured_output_schema:
        structured_output_base_model = json_schema_to_pydantic_model(structured_output_schema)


    if use_responses_api:
        if stream:
            return StreamOpenAIResponsesClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
                tools=tools,
                reasoning=reasoning,
                include=include,
                structured_output_base_model=structured_output_base_model,
            )
        else:
            return NonStreamOpenAIResponsesClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
                tools=tools,
                reasoning=reasoning,
                include=include,
                structured_output_base_model=structured_output_base_model,
            )
    else:
        if stream:
            return StreamOpenAIClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
                tools=tools,
                reasoning=reasoning,
            )
        else:
            return NonStreamOpenAIClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
                tools=tools,
                reasoning=reasoning,
            )
