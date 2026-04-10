from xai_sdk.chat import user, assistant, system, text, image, file

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type, get_filename_without_timestamp
from llm_bridge.type.message import Message, ContentType, Role
from llm_bridge.type.model_message.xai_message import XAIMessage, XAIContent


def create_unsupported_content(file_url: str, file_type: str, sub_type: str) -> XAIContent:
    return text(f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n")


async def convert_message_to_xai(message: Message) -> XAIMessage:
    contents: list[XAIContent] = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            contents.append(
                text(
                    content=content_item.data
                )
            )
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if file_type == "image":
                contents.append(
                    image(
                        image_url=file_url,
                        detail="high"
                    )
                )
            else:
                file_data, media_type = await media_processor.get_bytes_content_from_url(file_url)
                filename = get_filename_without_timestamp(file_url)
                contents.append(
                    file(
                        data=file_data,
                        filename=filename,
                        mime_type=media_type,
                    )
                )

    if message.role == Role.User:
        return user(*contents)
    elif message.role == Role.Assistant:
        return assistant(*contents)
    elif message.role == Role.System:
        return system(*contents)
    else:
        raise ValueError(f"Invalid role: {message.role}")
