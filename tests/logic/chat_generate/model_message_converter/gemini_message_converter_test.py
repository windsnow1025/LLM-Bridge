import pytest
from google.genai import types

from llm_bridge.logic.chat_generate.model_message_converter.gemini_message_converter import convert_message_to_gemini
from llm_bridge.type.message import Content, ContentType, Message, Role


@pytest.mark.asyncio
async def test_convert_message_to_gemini_user():
    message = Message(role=Role.User, contents=[Content(type=ContentType.Text, data="hello")])
    gemini_message = await convert_message_to_gemini(message)
    assert gemini_message == types.Content(parts=[types.Part.from_text(text="hello")], role="user")


@pytest.mark.asyncio
async def test_convert_message_to_gemini_assistant():
    message = Message(role=Role.Assistant, contents=[Content(type=ContentType.Text, data="hello")])
    gemini_message = await convert_message_to_gemini(message)
    assert gemini_message == types.Content(parts=[types.Part.from_text(text="hello")], role="model")


@pytest.mark.asyncio
async def test_convert_message_to_gemini_system():
    message = Message(role=Role.System, contents=[Content(type=ContentType.Text, data="hello")])
    gemini_message = await convert_message_to_gemini(message)
    assert gemini_message == types.Content(parts=[types.Part.from_text(text="hello")], role="user")
