from collections.abc import Callable

import pytest
from xai_sdk.chat import assistant, system, text, user

from llm_bridge.logic.chat_generate.model_message_converter.xai_message_converter import convert_message_to_xai
from llm_bridge.type.message import Content, ContentType, Message, Role
from llm_bridge.type.model_message.xai_message import XAIContent, XAIMessage


@pytest.mark.parametrize("role, create_message", [
    (Role.User, user),
    (Role.Assistant, assistant),
    (Role.System, system),
])
@pytest.mark.asyncio
async def test_convert_message_to_xai(role: Role, create_message: Callable[[XAIContent], XAIMessage]):
    message = Message(role=role, contents=[Content(type=ContentType.Text, data="hello")])
    xai_message = await convert_message_to_xai(message)
    assert xai_message == create_message(text(content="hello"))
