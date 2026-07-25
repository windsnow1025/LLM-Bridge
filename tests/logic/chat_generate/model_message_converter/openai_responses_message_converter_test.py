import pytest

from llm_bridge.logic.chat_generate.model_message_converter.openai_responses_message_converter import \
    convert_message_to_openai_responses
from llm_bridge.type.message import Content, ContentType, Message, Role
from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage


@pytest.mark.parametrize("role, expected", [
    (Role.User, {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}),
    (Role.Assistant, {"role": "assistant", "content": [{"type": "output_text", "text": "hello", "annotations": []}]}),
    (Role.System, {"role": "system", "content": [{"type": "input_text", "text": "hello"}]}),
])
@pytest.mark.asyncio
async def test_convert_message_to_openai_responses(role: Role, expected: OpenAIResponsesMessage):
    message = Message(role=role, contents=[Content(type=ContentType.Text, data="hello")])
    openai_message = await convert_message_to_openai_responses(message)
    assert openai_message == expected
