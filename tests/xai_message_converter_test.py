import pytest
from xai_sdk.chat import assistant, system, text, user

from llm_bridge.logic.chat_generate.model_message_converter.xai_message_converter import convert_message_to_xai
from llm_bridge.type.message import Content, ContentType, Message, Role


@pytest.mark.asyncio
async def test_convert_message_to_xai_user():
    message = Message(role=Role.User, contents=[Content(type=ContentType.Text, data="hello")])
    xai_message = await convert_message_to_xai(message)
    assert xai_message == user(text(content="hello"))


@pytest.mark.asyncio
async def test_convert_message_to_xai_assistant():
    message = Message(role=Role.Assistant, contents=[Content(type=ContentType.Text, data="hello")])
    xai_message = await convert_message_to_xai(message)
    assert xai_message == assistant(text(content="hello"))


@pytest.mark.asyncio
async def test_convert_message_to_xai_system():
    message = Message(role=Role.System, contents=[Content(type=ContentType.Text, data="hello")])
    xai_message = await convert_message_to_xai(message)
    assert xai_message == system(text(content="hello"))
