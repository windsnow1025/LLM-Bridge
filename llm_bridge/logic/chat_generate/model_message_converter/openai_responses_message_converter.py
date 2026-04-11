from openai.types.responses import ResponseInputTextParam, ResponseInputImageParam, ResponseOutputTextParam, \
    EasyInputMessageParam, ResponseOutputMessageParam, ResponseInputFileParam
# from openai.types.responses import ResponseInputAudioParam
# from openai.types.responses.response_input_audio_param import InputAudio
from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_process.file_type_checker import get_file_type, get_filename_without_timestamp
from llm_bridge.logic.message_process.message_processor import extract_file_as_text
# from llm_bridge.logic.message_preprocess.file_type_checker import get_file_extension
from llm_bridge.type.message import Message, ContentType, Role
from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage, \
    OpenAIResponsesContent, OpenAIResponsesRole


def create_unsupported_content(file_url: str, file_type: str, sub_type: str) -> ResponseInputTextParam:
    return ResponseInputTextParam(
        type="input_text",
        text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
    )


async def convert_input_content(message: Message) -> tuple[list[OpenAIResponsesContent], bool]:
    content: list[OpenAIResponsesContent] = []

    contains_pdf = False

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            content.append(ResponseInputTextParam(type="input_text", text=content_item.data))
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if file_type == "image":
                base64_image, media_type = await media_processor.get_base64_content_from_url(file_url)
                image_url = f"data:{media_type};base64,{base64_image}"
                content.append(ResponseInputImageParam(
                    type="input_image",
                    image_url=image_url,
                    detail="auto"
                ))
            elif sub_type == "pdf":
                contains_pdf = True
                file_data, _ = await media_processor.get_base64_content_from_url(file_url)
                content.append(ResponseInputFileParam(
                    type="input_file",
                    filename=get_filename_without_timestamp(file_url),
                    file_data=f"data:application/pdf;base64,{file_data}",
                ))
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
            elif file_type in ("text", "application"):
                extracted_text = await extract_file_as_text(file_url)
                content.append(ResponseInputTextParam(type="input_text", text=extracted_text))
            else:
                content.append(create_unsupported_content(file_url, file_type, sub_type))

    return content, contains_pdf


async def convert_output_content(message: Message) -> list[OpenAIResponsesContent]:
    content: list[OpenAIResponsesContent] = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            content.append(ResponseOutputTextParam(type="output_text", text=content_item.data, annotations=[]))

    return content


async def convert_message_to_openai_responses(message: Message) -> OpenAIResponsesMessage:
    if message.role == Role.Assistant:
        role: OpenAIResponsesRole = Role.Assistant
        output_content = await convert_output_content(message)
        return ResponseOutputMessageParam(role=role, content=output_content)
    else:
        input_content, contains_pdf = await convert_input_content(message)
        role: OpenAIResponsesRole
        if message.role == Role.System and contains_pdf:
            role = Role.User
        elif message.role == Role.System:
            role = Role.System
        else:
            role = Role.User
        return EasyInputMessageParam(role=role, content=input_content)
