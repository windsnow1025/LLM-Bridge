from google.genai import types

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type
from llm_bridge.type.message import Message, Role, ContentType
from llm_bridge.type.model_message.gemini_message import GeminiMessage, GeminiRole, GeminiContent


async def convert_message_to_gemini(message: Message) -> GeminiMessage:
    role: GeminiRole

    if message.role in (Role.User, Role.System):
        role = GeminiRole.User
    elif message.role == Role.Assistant:
        role = GeminiRole.Model
    else:
        raise ValueError(f"Invalid role: {message.role}")

    contents: list[GeminiContent] = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            contents.append(types.Part.from_text(text=content_item.data))
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if sub_type == "pdf" or file_type in ("image", "video", "audio"):
                file_data, media_type = await media_processor.get_bytes_content_from_url(file_url)
                if media_type == 'video/webm':
                    media_type = 'audio/webm'
                contents.append(types.Part.from_bytes(data=file_data, mime_type=media_type))
            else:
                text_content = types.Part.from_text(
                    text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
                )
                contents.append(text_content)

    return GeminiMessage(parts=contents, role=role)
