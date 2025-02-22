import pytest
from openai import AsyncOpenAI

from llm_bridge.client.implementations.gpt.non_stream_gpt_client import NonStreamGPTClient
from llm_bridge.logic.chat_generate.chat_client_factory import create_chat_client
from llm_bridge.type.model_message.gpt_message import GptMessage, TextContent
from llm_bridge.type.message import Message, Role


@pytest.fixture
def sample_messages():
    return [
        Message(role=Role.System, text="You are a helpful assistant.", files=[]),
        Message(role=Role.User, text="Hello", files=[])
    ]


@pytest.mark.asyncio
async def test_create_gpt_client_openai():
    messages = [Message(role=Role.User, text="Hello", files=[])]
    model = "gpt-4o"
    api_type = "OpenAI"
    temperature = 0
    stream = False
    api_keys = {
        "OPENAI_API_KEY": "test-key"
    }

    client = await create_chat_client(
        messages=messages,
        model=model,
        api_type=api_type,
        temperature=temperature,
        stream=stream,
        api_keys=api_keys
    )

    assert isinstance(client, NonStreamGPTClient)
    assert client.model == model
    assert client.temperature == temperature
    assert client.api_type == api_type
    assert isinstance(client.client, AsyncOpenAI)

    assert len(client.messages) == len(messages)
    assert isinstance(client.messages[0], GptMessage)
    assert client.messages[0].role == Role.User
    assert isinstance(client.messages[0].content, list)
    assert isinstance(client.messages[0].content[0], TextContent)
    assert client.messages[0].content[0].text == "Hello"
