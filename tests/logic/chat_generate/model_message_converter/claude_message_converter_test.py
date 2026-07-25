import pytest

from llm_bridge.logic.chat_generate.model_message_converter.claude_message_converter import convert_message_to_claude
from llm_bridge.type.message import Content, ContentType, Message, Role


@pytest.mark.parametrize("role, claude_role", [
    (Role.User, "user"),
    (Role.Assistant, "assistant"),
    (Role.System, "user"),
])
@pytest.mark.asyncio
async def test_convert_message_to_claude(role: Role, claude_role: str):
    message = Message(role=role, contents=[Content(type=ContentType.Text, data="hello")])
    claude_message = await convert_message_to_claude(message)
    assert claude_message == {"role": claude_role, "content": [{"type": "text", "text": "hello"}]}
