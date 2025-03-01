from llm_bridge.logic.message_preprocess import document_processor
from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type
from llm_bridge.type.message import Message, Role


async def preprocess_messages(messages: list[Message]) -> None:
    for message in messages:
        await extract_text_files_to_message(message)


async def extract_text_files_to_message(message: Message) -> None:
    indices_to_delete = []
    for i, file_url in enumerate(message.files[::-1]):
        file_type, sub_type = await get_file_type(file_url)
        if file_type != "text" and file_type != "application":
            continue
        filename = file_url.rsplit('/', 1)[-1].split('-', 1)[1]
        file_text = await document_processor.extract_text_from_file(file_url)
        message.text = f"{filename}: \n{file_text}\n{message.text}"
        indices_to_delete.append(len(message.files) - 1 - i)

    for index in sorted(indices_to_delete, reverse=True):
        del message.files[index]


def extract_system_messages(messages: list[Message]) -> str:
    system_message = ""
    indices_to_delete = []
    for i, message in enumerate(messages):
        if message.role != Role.System:
            continue
        system_message += f"{message.text}\n"
        indices_to_delete.append(i)

    for index in sorted(indices_to_delete):
        del messages[index]

    return system_message
