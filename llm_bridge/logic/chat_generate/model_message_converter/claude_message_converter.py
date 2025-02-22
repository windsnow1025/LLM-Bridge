from llm_bridge.logic.chat_generate import media_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type
from llm_bridge.type.model_message import claude_message
from llm_bridge.type.model_message.claude_message import ClaudeMessage
from llm_bridge.type.message import Message, Role


async def convert_message_to_claude(message: Message) -> ClaudeMessage:
    role = message.role
    text = message.text
    file_urls = message.files

    if role == Role.System:
        role = Role.User

    content = []

    if text:
        text_content = claude_message.TextContent(type="text", text=text)
        content.append(text_content)

    for file_url in file_urls:
        file_type, sub_type = get_file_type(file_url)
        if file_type == "image":
            image_contents = await media_processor.get_claude_image_content_from_url(file_url)
            content.append(image_contents)
            continue
        text_content = claude_message.TextContent(
            type="text",
            text=f"\n{file_url}: {file_type}/{sub_type} not supported by the current model.\n"
        )
        content.append(text_content)

    return ClaudeMessage(role=role, content=content)
