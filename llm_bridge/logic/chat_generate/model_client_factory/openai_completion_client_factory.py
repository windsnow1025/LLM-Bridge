import re

import openai
from fastapi import HTTPException
from openai import Omit
from openai.types.shared import ReasoningEffort

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
) -> StreamOpenAICompletionClient | NonStreamOpenAICompletionClient:
    if api_type == "OpenAI-GitHub":
        client = openai.AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=api_keys["GITHUB_API_KEY"],
        )
    else:
        raise HTTPException(status_code=500, detail="API Type not matched")

    openai_messages = await convert_messages_to_openai(messages)

    omit = Omit()
    reasoning_effort: ReasoningEffort | Omit = omit
    if re.match(r"gpt-5.*", model):
        if thought:
            reasoning_effort = "high"

    if stream:
        return StreamOpenAICompletionClient(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            api_type=api_type,
            client=client,
            reasoning_effort=reasoning_effort,
        )
    else:
        return NonStreamOpenAICompletionClient(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            api_type=api_type,
            client=client,
            reasoning_effort=reasoning_effort,
        )
