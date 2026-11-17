from google.genai.interactions import TextContentParam, DocumentContentParam, ImageContentParam, AudioContentParam, \
    VideoContentParam, UserInputStepParam, ModelOutputStepParam

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.chat_generate.model_message_converter.gemini_mime_types import GeminiDocumentMimeTypes, \
    GeminiImageMimeTypes, GeminiAudioMimeTypes, GeminiVideoMimeTypes
from llm_bridge.logic.message_process.file_type_checker import get_file_type
from llm_bridge.logic.message_process.message_processor import extract_file_as_text
from llm_bridge.type.message import Message, Role, ContentType
from llm_bridge.type.model_message.gemini_interactions_message import GeminiInteractionsMessage, \
    GeminiInteractionsContent


async def convert_message_to_gemini_interactions(message: Message) -> GeminiInteractionsMessage:
    contents: list[GeminiInteractionsContent] = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            contents.append(TextContentParam(type="text", text=content_item.data))
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            base64_data, media_type = await media_processor.get_base64_content_from_url(file_url)
            if media_type in GeminiDocumentMimeTypes:
                contents.append(DocumentContentParam(type="document", data=base64_data, mime_type=media_type))
            elif media_type in GeminiImageMimeTypes:
                contents.append(ImageContentParam(type="image", data=base64_data, mime_type=media_type))
            elif media_type in GeminiAudioMimeTypes:
                contents.append(AudioContentParam(type="audio", data=base64_data, mime_type=media_type))
            elif media_type in GeminiVideoMimeTypes:
                contents.append(VideoContentParam(type="video", data=base64_data, mime_type=media_type))
            else:
                file_type, sub_type = await get_file_type(file_url)
                if file_type in ("text", "application"):
                    extracted_text = await extract_file_as_text(file_url)
                    contents.append(TextContentParam(type="text", text=extracted_text))
                else:
                    contents.append(TextContentParam(
                        type="text",
                        text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
                    ))

    if message.role in (Role.User, Role.System):
        return UserInputStepParam(type="user_input", content=contents)
    elif message.role == Role.Assistant:
        return ModelOutputStepParam(type="model_output", content=contents)
    else:
        raise ValueError(f"Invalid role: {message.role}")
