from llm_bridge.type.message import Content, ContentType, Message, Role


def create_text_message(role: Role, text: str) -> Message:
    return Message(role=role, contents=[Content(type=ContentType.Text, data=text)])
