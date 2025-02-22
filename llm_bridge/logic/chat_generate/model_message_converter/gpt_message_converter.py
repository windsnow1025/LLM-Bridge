from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, \
    ChatCompletionContentPartInputAudioParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type
from llm_bridge.type.model_message import gpt_message
from llm_bridge.type.model_message.gpt_message import GptMessage
from llm_bridge.type.message import Message


async def convert_message_to_gpt(message: Message) -> GptMessage:
    role = message.role
    text = message.text
    file_urls = message.files

    content = []

    if text:
        text_content = ChatCompletionContentPartTextParam(type="text", text=text)
        content.append(text_content)

    for file_url in file_urls:
        file_type, sub_type = get_file_type(file_url)
        if file_type == "image":
            image_url = await media_processor.get_gpt_image_content_from_url(file_url)
            image_content = ChatCompletionContentPartImageParam(
                type="image_url",
                image_url=ImageURL(url=image_url)
            )
            content.append(image_content)
        elif file_type == "audio":
            encoded_string = await media_processor.get_gpt_audio_content_from_url(file_url)
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

    return GptMessage(role=role, content=content)
