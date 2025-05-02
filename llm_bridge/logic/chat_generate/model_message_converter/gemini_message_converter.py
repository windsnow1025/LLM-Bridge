from google.genai import types

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type
from llm_bridge.type.model_message.gemini_message import GeminiMessage, GeminiRole
from llm_bridge.type.message import Message, Role, ContentType, Content


async def convert_message_to_gemini(message: Message) -> GeminiMessage:
    role = message.role

    if role == Role.System:
        role = GeminiRole.User
    if role == Role.Assistant:
        role = GeminiRole.Model

    parts = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            parts.append(types.Part.from_text(text=content_item.data))
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if sub_type == "pdf" or file_type == "image":
                pdf_bytes, media_type = await media_processor.get_raw_content_from_url(file_url)
                parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type=media_type))
            elif file_type == "audio":
                audio_bytes, media_type = await media_processor.get_gemini_audio_content_from_url(file_url)
                parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=media_type))
            else:
                text_content = types.Part.from_text(
                    text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
                )
                parts.append(text_content)

    return GeminiMessage(parts=parts, role=role)
