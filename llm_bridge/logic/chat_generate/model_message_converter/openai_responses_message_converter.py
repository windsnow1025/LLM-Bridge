from openai.types.responses import ResponseInputTextParam, ResponseInputImageParam, ResponseOutputTextParam, \
    ResponseInputContentParam, EasyInputMessageParam, ResponseOutputMessageParam, ResponseInputFileParam
# from openai.types.responses import ResponseInputAudioParam
# from openai.types.responses.response_input_audio_param import InputAudio
from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type, get_filename_without_timestamp
# from llm_bridge.logic.message_preprocess.file_type_checker import get_file_extension
from llm_bridge.type.message import Message, ContentType, Role
from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage


def create_unsupported_content(file_url: str, file_type: str, sub_type: str) -> ResponseInputTextParam:
    return ResponseInputTextParam(
        type="input_text",
        text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
    )


async def convert_message_to_openai_responses(message: Message) -> OpenAIResponsesMessage:
    role = message.role
    content: list[ResponseInputContentParam | ResponseOutputTextParam] = []
    contains_pdf = False

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            if role == Role.Assistant:
                text_content = ResponseOutputTextParam(type="output_text", text=content_item.data, annotations=[])
            else:
                text_content = ResponseInputTextParam(type="input_text", text=content_item.data)
            content.append(text_content)
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if file_type == "image":
                base64_image, media_type = await media_processor.get_base64_content_from_url(file_url)
                image_url = f"data:{media_type};base64,{base64_image}"
                image_content = ResponseInputImageParam(
                    type="input_image",
                    image_url=image_url,
                    detail="auto"
                )
                content.append(image_content)
            elif sub_type == "pdf":
                contains_pdf = True
                file_data, _ = await media_processor.get_base64_content_from_url(file_url)
                pdf_content = ResponseInputFileParam(
                    type="input_file",
                    filename=get_filename_without_timestamp(file_url),
                    file_data=f"data:application/pdf;base64,{file_data}",
                )
                content.append(pdf_content)
            # Audio Input not supported in Responses API
            # elif file_type == "audio":
            #     audio_format = get_file_extension(file_url)
            #     if audio_format in ("wav", "mp3"):
            #         file_data, _ = await media_processor.get_base64_content_from_url(file_url)
            #         audio_content = ResponseInputAudioParam(
            #             type="input_audio",
            #             input_audio=InputAudio(data=file_data, format=audio_format)
            #         )
            #         content.append(audio_content)
            #     else:
            #         content.append(create_unsupported_content(file_url, file_type, sub_type))
            else:
                content.append(create_unsupported_content(file_url, file_type, sub_type))

    # Force system role to user if the message contains a PDF
    if role == Role.System and contains_pdf:
        role = Role.User

    if role in (Role.User, Role.System):
        return EasyInputMessageParam(role=role.value, content=content)
    else:
        return ResponseOutputMessageParam(role=role.value, content=content)
