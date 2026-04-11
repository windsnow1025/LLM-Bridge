import xai_sdk
from xai_sdk.chat import ReasoningEffort

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
) -> StreamXAIClient | NonStreamXAIClient:
    client = xai_sdk.AsyncClient(
        api_key=api_key,
    )

    xai_messages = await convert_messages_to_xai(messages)

    if stream:
        return StreamXAIClient(
            model=model,
            messages=xai_messages,
            temperature=temperature,
            client=client
        )
    else:
        return NonStreamXAIClient(
            model=model,
            messages=xai_messages,
            temperature=temperature,
            client=client
        )
