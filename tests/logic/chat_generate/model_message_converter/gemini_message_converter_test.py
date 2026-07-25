import pytest
from google.genai import types

from llm_bridge.logic.chat_generate.model_message_converter.gemini_message_converter import convert_message_to_gemini
from llm_bridge.type.message import Role
from tests.logic.chat_generate.model_message_converter.message_factory import create_text_message


@pytest.mark.parametrize("role, gemini_role", [
    (Role.User, "user"),
    (Role.Assistant, "model"),
    (Role.System, "user"),
])
@pytest.mark.asyncio
async def test_convert_message_to_gemini(role: Role, gemini_role: str):
    message = create_text_message(role, "hello")
    gemini_message = await convert_message_to_gemini(message)
    assert gemini_message == types.Content(parts=[types.Part.from_text(text="hello")], role=gemini_role)
