import pytest

from llm_bridge.logic.chat_generate.chat_client_factory import create_chat_client
from llm_bridge.type.message import Message, Role, Content, ContentType


@pytest.fixture
def sample_messages():
    return [
        Message(role=Role.System, contents=[
            Content(type=ContentType.Text, data="You are a helpful assistant.")
        ]),
        Message(role=Role.User, contents=[
            Content(type=ContentType.Text, data="Hello")
        ])
    ]


@pytest.mark.asyncio
async def test_create_gpt_client_openai():
    messages = [Message(role=Role.User, contents=[
        Content(type=ContentType.Text, data="Hello")
    ])]
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

    assert True
