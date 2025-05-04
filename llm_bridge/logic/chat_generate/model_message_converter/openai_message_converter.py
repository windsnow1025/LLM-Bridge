from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, \
    ChatCompletionContentPartInputAudioParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type
from llm_bridge.type.message import Message, ContentType
from llm_bridge.type.model_message.openai_message import OpenAIMessage


async def convert_message_to_openai(message: Message) -> OpenAIMessage:
    role = message.role
    content = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            text_content = ChatCompletionContentPartTextParam(type="text", text=content_item.data)
            content.append(text_content)
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if file_type == "image":
                image_url = await media_processor.get_openai_image_content_from_url(file_url)
                image_content = ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(url=image_url)
                )
                content.append(image_content)
            elif file_type == "audio":
                encoded_string, _ = await media_processor.get_encoded_content_from_url(file_url)
                audio_content = ChatCompletionContentPartInputAudioParam(
                    type="input_audio",
                    input_audio=InputAudio(data=encoded_string, format=sub_type)
                )
                content.append(audio_content)
            else:
                text_content = ChatCompletionContentPartTextParam(
                    type="text",
                    text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
                )
                content.append(text_content)

    return OpenAIMessage(role=role, content=content)
