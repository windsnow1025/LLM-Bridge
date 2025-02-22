from google.genai import types

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type
from llm_bridge.type.model_message.gemini_message import GeminiMessage
from llm_bridge.type.message import Message, Role


async def convert_message_to_gemini(message: Message) -> GeminiMessage:
    role = message.role
    text = message.text
    file_urls = message.files

    if role == Role.System:
        role = Role.User
    if role == Role.Assistant:
        role = Role.Model

    parts = [types.Part.from_text(text=text)]

    for file_url in file_urls:
        file_type, sub_type = get_file_type(file_url)
        if file_type == "image":
            img_bytes, media_type = await media_processor.get_gemini_image_content_from_url(file_url)
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=media_type))
        elif file_type == "audio":
            audio_bytes, media_type = await media_processor.get_gemini_audio_content_from_url(file_url)
            parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=media_type))
        else:
            text_content = types.Part.from_text(
                text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
            )
            parts.append(text_content)


    return GeminiMessage(parts=parts, role=role)
