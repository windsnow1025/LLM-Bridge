from anthropic.types import TextBlockParam, ImageBlockParam, DocumentBlockParam, Base64ImageSourceParam, \
    Base64PDFSourceParam

from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_process.file_type_checker import get_file_type
from llm_bridge.logic.message_process.message_processor import extract_file_as_text
from llm_bridge.type.message import Message, Role, ContentType
from llm_bridge.type.model_message.claude_message import ClaudeMessage, ClaudeRole, ClaudeContent


def create_unsupported_content(file_url: str, file_type: str, sub_type: str) -> TextBlockParam:
    return TextBlockParam(
        type="text",
        text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
    )


async def convert_message_to_claude(message: Message) -> ClaudeMessage:
    role: ClaudeRole

    if message.role in (Role.System, Role.User):
        role = ClaudeRole.User
    elif message.role == Role.Assistant:
        role = ClaudeRole.Assistant
    else:
        raise ValueError(f"Invalid role: {message.role}")

    claude_content: list[ClaudeContent] = []

    for content_item in message.contents:
        if content_item.type == ContentType.Text:
            text_content = TextBlockParam(type="text", text=content_item.data)
            claude_content.append(text_content)
        elif content_item.type == ContentType.File:
            file_url = content_item.data
            file_type, sub_type = await get_file_type(file_url)
            if file_type == "image":
                base64_image, media_type = await media_processor.get_base64_content_from_url(file_url)
                if media_type in ("image/jpeg", "image/png", "image/gif", "image/webp"):
                    image_content = ImageBlockParam(
                        type="image",
                        source=Base64ImageSourceParam(
                            type="base64",
                            media_type=media_type,
                            data=base64_image
                        )
                    )
                    claude_content.append(image_content)
                else:
                    claude_content.append(create_unsupported_content(file_url, file_type, sub_type))
            elif sub_type == "pdf":
                file_data, media_type = await media_processor.get_base64_content_from_url(file_url)
                if media_type == "application/pdf":
                    pdf_content = DocumentBlockParam(
                        type="document",
                        source=Base64PDFSourceParam(
                            type="base64",
                            media_type=media_type,
                            data=file_data
                        ),
                    )
                    claude_content.append(pdf_content)
                else:
                    claude_content.append(create_unsupported_content(file_url, file_type, sub_type))
            elif file_type in ("text", "application"):
                extracted_text = await extract_file_as_text(file_url)
                claude_content.append(TextBlockParam(type="text", text=extracted_text))
            else:
                claude_content.append(create_unsupported_content(file_url, file_type, sub_type))

    return ClaudeMessage(role=role, content=claude_content)
