import openai.lib.azure

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
    client = None
    if api_type == "OpenAI":
        client = openai.AsyncOpenAI(
            api_key=api_keys["OPENAI_API_KEY"],
        )
    elif api_type == "Azure":
        client = openai.lib.azure.AsyncAzureOpenAI(
            api_version="2025-01-01-preview",
            azure_endpoint=api_keys["AZURE_API_BASE"],
            api_key=api_keys["AZURE_API_KEY"],
        )
    elif api_type == "GitHub":
        client = openai.AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=api_keys["GITHUB_API_KEY"],
        )
    elif api_type == "Grok":
        client = openai.AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_keys["XAI_API_KEY"],
        )


    if api_type == "OpenAI":
        openai_messages = await convert_messages_to_openai_responses(messages)
    else:
        openai_messages = await convert_messages_to_openai(messages)


    if api_type == "OpenAI":
        if stream:
            return StreamOpenAIResponsesClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
            )
        else:
            return NonStreamOpenAIResponsesClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
            )
    else:
        if stream:
            return StreamOpenAIClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
            )
        else:
            return NonStreamOpenAIClient(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                api_type=api_type,
                client=client,
            )
