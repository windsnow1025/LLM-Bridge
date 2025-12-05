import pytest

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
async def test_placeholder():
    assert True
