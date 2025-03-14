import pytest

from llm_bridge.logic.message_preprocess.message_preprocessor import extract_system_messages
from llm_bridge.type.message import Message, Role, Content, ContentType


@pytest.fixture
def sample_messages():
    return [
        Message(role=Role.System, contents=[
            Content(type=ContentType.Text, data="You are a helpful assistant.")
        ]),
        Message(role=Role.User, contents=[
            Content(type=ContentType.Text, data="Hello")
        ])
    ]

def test_extract_system_messages(sample_messages):
    extracted_text = extract_system_messages(sample_messages)

    assert extracted_text == "You are a helpful assistant.\n"

    assert len(sample_messages) == 1
    assert sample_messages[0].role == Role.User
    assert sample_messages[0].contents[0].type == ContentType.Text
    assert sample_messages[0].contents[0].data == "Hello"
