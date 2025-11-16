import re

import openai
from fastapi import HTTPException
from openai.types import Reasoning
from openai.types.responses import WebSearchToolParam
from openai.types.responses.tool_param import CodeInterpreter, CodeInterpreterContainerCodeInterpreterToolAuto, \
    ImageGeneration

from llm_bridge.client.implementations.openai.non_stream_openai_client import NonStreamOpenAIClient
from llm_bridge.client.implementations.openai.non_stream_openai_responses_client import NonStreamOpenAIResponsesClient
from llm_bridge.client.implementations.openai.steam_openai_responses_client import StreamOpenAIResponsesClient
from llm_bridge.client.implementations.openai.stream_openai_client import StreamOpenAIClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_openai_responses, \
    convert_messages_to_openai
from llm_bridge.type.message import Message


async def create_openai_client(
        messages: list[Message],
        model: str,
        api_type: str,
        temperature: float,
        stream: bool,
        api_keys: dict
):
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

    if api_type in ("OpenAI", "OpenAI-Azure", "Grok"):
        use_responses_api = True
    else:
        use_responses_api = False

    if use_responses_api:
        openai_messages = await convert_messages_to_openai_responses(messages)
    else:
        openai_messages = await convert_messages_to_openai(messages)

    tools = []
    reasoning = None

    if model not in ["gpt-5-chat-latest", "gpt-5-pro"]:
        tools.append(
            CodeInterpreter(
                type="code_interpreter",
                container=CodeInterpreterContainerCodeInterpreterToolAuto(type="auto")
            )
        )
    if model not in ["gpt-5-chat-latest"]:
        tools.append(
            WebSearchToolParam(
                type="web_search",
                search_context_size="high",
            )
        )
    if re.match(r"^o\d", model) or (re.match(r"gpt-5.*", model) and model != "gpt-5-chat-latest"):
        temperature = 1
    if re.match(r"gpt-5.*", model) and model != "gpt-5-chat-latest":
        reasoning = Reasoning(
            effort="high",
            summary="auto",
        )
        tools.append(
            ImageGeneration(
                type="image_generation",
            )
        )

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
