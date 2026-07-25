import pytest

from llm_bridge.logic.chat_generate.model_message_converter.openai_responses_message_converter import \
    convert_message_to_openai_responses
from llm_bridge.type.message import Content, ContentType, Message, Role


@pytest.mark.asyncio
async def test_convert_message_to_openai_responses_user():
    message = Message(role=Role.User, contents=[Content(type=ContentType.Text, data="hello")])
    openai_message = await convert_message_to_openai_responses(message)
    assert openai_message == {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}


@pytest.mark.asyncio
async def test_convert_message_to_openai_responses_assistant():
    message = Message(role=Role.Assistant, contents=[Content(type=ContentType.Text, data="hello")])
    openai_message = await convert_message_to_openai_responses(message)
    assert openai_message == {"role": "assistant", "content": [{"type": "output_text", "text": "hello", "annotations": []}]}


@pytest.mark.asyncio
async def test_convert_message_to_openai_responses_system():
    message = Message(role=Role.System, contents=[Content(type=ContentType.Text, data="hello")])
    openai_message = await convert_message_to_openai_responses(message)
    assert openai_message == {"role": "system", "content": [{"type": "input_text", "text": "hello"}]}
