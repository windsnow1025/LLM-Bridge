import pytest

from llm_bridge.logic.chat_generate.model_message_converter.openai_completion_message_converter import \
    convert_message_to_openai_completion
from llm_bridge.type.message import Content, ContentType, Message, Role


@pytest.mark.parametrize("role, openai_role", [
    (Role.User, "user"),
    (Role.Assistant, "assistant"),
    (Role.System, "system"),
])
@pytest.mark.asyncio
async def test_convert_message_to_openai_completion(role: Role, openai_role: str):
    message = Message(role=role, contents=[Content(type=ContentType.Text, data="hello")])
    openai_message = await convert_message_to_openai_completion(message)
    assert openai_message == {"role": openai_role, "content": [{"type": "text", "text": "hello"}]}
