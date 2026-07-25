import pytest

from llm_bridge.logic.chat_generate.model_message_converter.claude_message_converter import convert_message_to_claude
from llm_bridge.type.message import Content, ContentType, Message, Role


@pytest.mark.asyncio
async def test_convert_message_to_claude_user():
    message = Message(role=Role.User, contents=[Content(type=ContentType.Text, data="hello")])
    claude_message = await convert_message_to_claude(message)
    assert claude_message == {"role": "user", "content": [{"type": "text", "text": "hello"}]}


@pytest.mark.asyncio
async def test_convert_message_to_claude_assistant():
    message = Message(role=Role.Assistant, contents=[Content(type=ContentType.Text, data="hello")])
    claude_message = await convert_message_to_claude(message)
    assert claude_message == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}


@pytest.mark.asyncio
async def test_convert_message_to_claude_system():
    message = Message(role=Role.System, contents=[Content(type=ContentType.Text, data="hello")])
    claude_message = await convert_message_to_claude(message)
    assert claude_message == {"role": "user", "content": [{"type": "text", "text": "hello"}]}
