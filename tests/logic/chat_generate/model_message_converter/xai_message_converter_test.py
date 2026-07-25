from collections.abc import Callable

import pytest
from xai_sdk.chat import assistant, system, text, user

from llm_bridge.logic.chat_generate.model_message_converter.xai_message_converter import convert_message_to_xai
from llm_bridge.type.message import Role
from llm_bridge.type.model_message.xai_message import XAIContent, XAIMessage
from tests.logic.chat_generate.model_message_converter.message_factory import create_text_message


@pytest.mark.parametrize("role, create_message", [
    (Role.User, user),
    (Role.Assistant, assistant),
    (Role.System, system),
])
@pytest.mark.asyncio
async def test_convert_message_to_xai(role: Role, create_message: Callable[[XAIContent], XAIMessage]):
    message = create_text_message(role, "hello")
    xai_message = await convert_message_to_xai(message)
    assert xai_message == create_message(text(content="hello"))
