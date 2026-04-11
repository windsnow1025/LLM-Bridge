from google.genai import types

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_process.file_type_checker import get_file_type
from llm_bridge.logic.message_process.message_processor import extract_file_as_text
from llm_bridge.type.message import Message, Role, ContentType
from llm_bridge.type.model_message.gemini_message import GeminiMessage, GeminiRole, GeminiContent

# https://ai.google.dev/gemini-api/docs/file-input-methods
GeminiSupportedMimeTypes = {
    "text/html",
    "text/css",
    "text/plain",
    "text/xml",
    "text/csv",
    "text/rtf",
    "text/javascript",

    "application/json",
    "application/pdf",

    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",

    # https://ai.google.dev/gemini-api/docs/audio#supported-formats
    "audio/wav",
    "audio/mp3",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "audio/flac",

    # https://ai.google.dev/gemini-api/docs/video-understanding#supported-formats
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
}


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
            file_data, media_type = await media_processor.get_bytes_content_from_url(file_url)
            if media_type in GeminiSupportedMimeTypes:
                contents.append(types.Part.from_bytes(data=file_data, mime_type=media_type))
            else:
                file_type, sub_type = await get_file_type(file_url)
                if file_type in ("text", "application"):
                    extracted_text = await extract_file_as_text(file_url)
                    contents.append(types.Part.from_text(text=extracted_text))
                else:
                    text_content = types.Part.from_text(
                        text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
                    )
                    contents.append(text_content)

    return GeminiMessage(parts=contents, role=role)
