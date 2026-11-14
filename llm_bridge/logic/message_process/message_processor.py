from llm_bridge.logic.message_process import document_processor
from llm_bridge.logic.message_process.file_type_checker import get_file_type, get_filename_without_timestamp
from llm_bridge.type.message import Message, Role, ContentType


async def extract_file_as_text(file_url: str) -> str:
    filename = get_filename_without_timestamp(file_url)
    file_text = await document_processor.extract_text_from_file(file_url)
    return f"<file name=\"{filename}\">\n{file_text}\n</file>"


def extract_system_message(messages: list[Message]) -> str:
    if not messages:
        return ""

    message = messages[0]
    if message.role != Role.System:
        return ""
    if any(content_item.type == ContentType.File for content_item in message.contents):
        return ""

    system_message = ""
    for content_item in message.contents:
        system_message += f"{content_item.data}\n"

    del messages[0]

    return system_message


async def contains_audio_or_video(messages: list[Message]) -> bool:
    for message in messages:
        for content_item in message.contents:
            if content_item.type != ContentType.File:
                continue
            file_type, _ = await get_file_type(content_item.data)
            if file_type in ("audio", "video"):
                return True
    return False
