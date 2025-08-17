from llm_bridge.logic.message_preprocess import document_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type, get_file_name
from llm_bridge.type.message import Message, Role, Content, ContentType


async def preprocess_messages(messages: list[Message], api_type: str) -> None:
    for message in messages:
        await extract_text_files_to_message(message, api_type)


async def extract_text_files_to_message(message: Message, api_type: str) -> None:
    for i in range(len(message.contents) - 1, -1, -1):
        content_item = message.contents[i]
        
        if content_item.type != ContentType.File:
            continue
            
        file_url = content_item.data
        file_type, sub_type = await get_file_type(file_url)
        
        if file_type != "text" and file_type != "application":
            continue

        if sub_type == "pdf" and api_type in ("OpenAI", "OpenAI-Azure", "Gemini-Free", "Gemini-Paid", "Claude"):
            continue
            
        filename = get_file_name(file_url)
        file_text = await document_processor.extract_text_from_file(file_url)
        
        message.contents[i] = Content(
            type=ContentType.Text,
            data=f"{filename}: \n{file_text}\n"
        )


def extract_system_messages(messages: list[Message]) -> str:
    system_message = ""
    indices_to_delete = []
    
    for i, message in enumerate(messages):
        if message.role != Role.System:
            continue
            
        # Concatenate all text content from the system message
        for content_item in message.contents:
            if content_item.type == ContentType.Text:
                system_message += f"{content_item.data}\n"
                
        indices_to_delete.append(i)

    for index in sorted(indices_to_delete, reverse=True):
        del messages[index]

    return system_message
