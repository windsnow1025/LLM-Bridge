import tiktoken
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.openai_message import OpenAIMessage

from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage


def count_openai_input_tokens(messages: list[OpenAIMessage]) -> int:
    text = ''
    file_count = 0

    for message in messages:
        for content in message.content:
            if content['type'] == "text":
                text += content['text']
            elif content['type'] in ("image_url", "input_audio"):
                file_count += 1

    return num_tokens_from_text(text) + file_count * 1000


def count_openai_responses_input_tokens(messages: list[OpenAIResponsesMessage]) -> int:
    text = ''
    file_count = 0

    for message in messages:
        for content in message['content']:
            if content['type'] in ("output_text", "input_text"):
                text += content['text']
            elif content['type'] in ("input_image", "input_file"):
                file_count += 1

    return num_tokens_from_text(text) + file_count * 1000


def count_openai_output_tokens(chat_response: ChatResponse) -> int:
    text = chat_response.text
    file_count = len(chat_response.files)

    return num_tokens_from_text(text) + file_count * 1000


def num_tokens_from_text(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
