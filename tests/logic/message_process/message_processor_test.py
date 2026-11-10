from llm_bridge.logic.message_process.message_processor import extract_system_message
from llm_bridge.type.message import Content, ContentType, Message, Role
from tests.logic.message_factory import create_text_message


def test_extract_system_message_empty():
    messages = []

    assert extract_system_message(messages) == ""
    assert messages == []


def test_extract_system_message_text():
    system_message = create_text_message(Role.System, "You are a helpful assistant.")
    user_message = create_text_message(Role.User, "Hello")
    messages = [system_message, user_message]

    assert extract_system_message(messages) == "You are a helpful assistant.\n"
    assert messages == [user_message]


def test_extract_system_message_file():
    system_message = Message(role=Role.System, contents=[
        Content(type=ContentType.Text, data="You are a helpful assistant."),
        Content(type=ContentType.File, data="https://example.com/1767243600000-document.pdf"),
    ])
    user_message = create_text_message(Role.User, "Hello")
    messages = [system_message, user_message]

    assert extract_system_message(messages) == ""
    assert messages == [system_message, user_message]


def test_extract_system_message_not_first():
    user_message = create_text_message(Role.User, "Hello")
    system_message = create_text_message(Role.System, "You are a helpful assistant.")
    messages = [user_message, system_message]

    assert extract_system_message(messages) == ""
    assert messages == [user_message, system_message]
