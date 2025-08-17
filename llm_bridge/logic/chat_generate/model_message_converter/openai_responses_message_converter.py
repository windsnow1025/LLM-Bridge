from openai.types.responses import ResponseInputTextParam, ResponseInputImageParam, ResponseOutputTextParam, \
    ResponseInputContentParam, EasyInputMessageParam, ResponseOutputMessageParam, ResponseInputFileParam

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type, get_file_name
from llm_bridge.type.message import Message, ContentType
from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage


async def convert_message_to_openai_responses(message: Message) -> OpenAIResponsesMessage:
    role = message.role.value
    content: list[ResponseInputContentParam | ResponseOutputTextParam] = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            if role == "assistant":
                text_content = ResponseOutputTextParam(type="output_text", text=content_item.data)
            else:
                text_content = ResponseInputTextParam(type="input_text", text=content_item.data)
            content.append(text_content)
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if file_type == "image":
                image_url = await media_processor.get_openai_image_content_from_url(file_url)
                image_content = ResponseInputImageParam(
                    type="input_image",
                    image_url=image_url,
                    detail="auto"
                )
                content.append(image_content)
            elif sub_type == "pdf":
                file_data, _ = await media_processor.get_encoded_content_from_url(file_url)
                pdf_content = ResponseInputFileParam(
                    type="input_file",
                    filename=get_file_name(file_url),
                    file_data=f"data:application/pdf;base64,{file_data}",
                )
                content.append(pdf_content)
            # TODO: Responses API is currently unsupported for audio input
            # elif file_type == "audio":
            #     encoded_string = await media_processor.get_gpt_audio_content_from_url(file_url)
            #     audio_content = ChatCompletionContentPartInputAudioParam(
            #         type="input_audio",
            #         input_audio=InputAudio(data=encoded_string, format=sub_type)
            #     )
            #     content.append(audio_content)
            else:
                text_content = ResponseInputTextParam(
                    type="input_text",
                    text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
                )
                content.append(text_content)

    if role in ("user", "system"):
        return EasyInputMessageParam(role=role, content=content)
    else:
        return ResponseOutputMessageParam(role=role, content=content)
