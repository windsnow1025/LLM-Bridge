from llm_bridge.logic.message_process.message_processor import extract_system_messages
from llm_bridge.type.message import Content, ContentType, Message, Role
from tests.logic.message_factory import create_text_message


def test_extract_system_messages_text():
    system_message = create_text_message(Role.System, "You are a helpful assistant.")
    user_message = create_text_message(Role.User, "Hello")
    messages = [system_message, user_message]

    assert extract_system_messages(messages) == "You are a helpful assistant.\n"
    assert messages == [user_message]


def test_extract_system_messages_file():
    system_message = Message(role=Role.System, contents=[
        Content(type=ContentType.Text, data="You are a helpful assistant."),
        Content(type=ContentType.File, data="https://example.com/1767243600000-document.pdf"),
    ])
    user_message = create_text_message(Role.User, "Hello")
    messages = [system_message, user_message]

    assert extract_system_messages(messages) == ""
    assert messages == [system_message, user_message]
