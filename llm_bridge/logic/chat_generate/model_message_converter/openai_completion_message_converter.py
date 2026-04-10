from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, \
    ChatCompletionContentPartInputAudioParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.chat.chat_completion_content_part_param import File, FileFile

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_process.file_type_checker import get_file_type, get_file_extension, \
    get_filename_without_timestamp
from llm_bridge.logic.message_process.message_preprocessor import extract_file_as_text
from llm_bridge.type.message import Message, ContentType, Role
from llm_bridge.type.model_message.openai_completion_message import OpenAICompletionMessage, OpenAICompletionContent


def create_unsupported_content(file_url: str, file_type: str, sub_type: str) -> ChatCompletionContentPartTextParam:
    return ChatCompletionContentPartTextParam(
        type="text",
        text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
    )


async def convert_message_to_openai_completion(message: Message) -> OpenAICompletionMessage:
    content: list[OpenAICompletionContent] = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            text_content = ChatCompletionContentPartTextParam(type="text", text=content_item.data)
            content.append(text_content)
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if file_type == "image":
                base64_image, media_type = await media_processor.get_base64_content_from_url(file_url)
                image_url = f"data:{media_type};base64,{base64_image}"
                image_content = ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(url=image_url)
                )
                content.append(image_content)
            elif file_type == "audio":
                audio_format = get_file_extension(file_url)
                if audio_format in ("wav", "mp3"):
                    encoded_string, _ = await media_processor.get_base64_content_from_url(file_url)
                    audio_content = ChatCompletionContentPartInputAudioParam(
                        type="input_audio",
                        input_audio=InputAudio(data=encoded_string, format=audio_format)
                    )
                    content.append(audio_content)
                else:
                    content.append(create_unsupported_content(file_url, file_type, sub_type))
            elif sub_type == "pdf":
                base64_data, media_type = await media_processor.get_base64_content_from_url(file_url)
                filename = get_filename_without_timestamp(file_url)
                file_content = File(
                    type="file",
                    file=FileFile(
                        file_data=f"data:{media_type};base64,{base64_data}",
                        filename=filename,
                    ),
                )
                content.append(file_content)
            elif file_type in ("text", "application"):
                extracted_text = await extract_file_as_text(file_url)
                content.append(ChatCompletionContentPartTextParam(type="text", text=extracted_text))
            else:
                content.append(create_unsupported_content(file_url, file_type, sub_type))

    if message.role == Role.System:
        return ChatCompletionSystemMessageParam(role=message.role, content=content)
    elif message.role == Role.Assistant:
        return ChatCompletionAssistantMessageParam(role=message.role, content=content)
    elif message.role == Role.User:
        return ChatCompletionUserMessageParam(role=message.role, content=content)
    else:
        raise ValueError(f"Invalid role: {message.role}")
