import pytest

from llm_bridge.logic.chat_generate.model_message_converter.gemini_interactions_message_converter import \
    convert_message_to_gemini_interactions
from llm_bridge.type.message import Role
from tests.logic.message_factory import create_text_message


@pytest.mark.parametrize("role, step_type", [
    (Role.User, "user_input"),
    (Role.Assistant, "model_output"),
    (Role.System, "user_input"),
])
@pytest.mark.asyncio
async def test_convert_message_to_gemini_interactions(role: Role, step_type: str):
    message = create_text_message(role, "hello")
    gemini_message = await convert_message_to_gemini_interactions(message)
    assert gemini_message == {"type": step_type, "content": [{"type": "text", "text": "hello"}]}
