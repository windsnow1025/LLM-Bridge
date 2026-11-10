from llm_bridge.logic.message_process import document_processor
from llm_bridge.logic.message_process.file_type_checker import get_filename_without_timestamp
from llm_bridge.type.message import Message, Role, ContentType


async def extract_file_as_text(file_url: str) -> str:
    filename = get_filename_without_timestamp(file_url)
    file_text = await document_processor.extract_text_from_file(file_url)
    return f"<file name=\"{filename}\">\n{file_text}\n</file>"


def extract_system_messages(messages: list[Message]) -> str:
    system_message = ""
    indices_to_delete = []
    
    for i, message in enumerate(messages):
        if message.role != Role.System:
            continue
        if any(content_item.type == ContentType.File for content_item in message.contents):
            continue

        for content_item in message.contents:
            system_message += f"{content_item.data}\n"

        indices_to_delete.append(i)

    for index in sorted(indices_to_delete, reverse=True):
        del messages[index]

    return system_message
