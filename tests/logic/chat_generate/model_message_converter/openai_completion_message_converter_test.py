import pytest

from llm_bridge.logic.chat_generate.model_message_converter.openai_completion_message_converter import \
    convert_message_to_openai_completion
from llm_bridge.type.message import Role
from tests.logic.chat_generate.model_message_converter.message_factory import create_text_message


@pytest.mark.parametrize("role, openai_role", [
    (Role.User, "user"),
    (Role.Assistant, "assistant"),
    (Role.System, "system"),
])
@pytest.mark.asyncio
async def test_convert_message_to_openai_completion(role: Role, openai_role: str):
    message = create_text_message(role, "hello")
    openai_message = await convert_message_to_openai_completion(message)
    assert openai_message == {"role": openai_role, "content": [{"type": "text", "text": "hello"}]}
