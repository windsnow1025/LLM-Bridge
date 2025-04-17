from llm_bridge.logic.chat_generate.model_message_converter.claude_message_converter import convert_message_to_claude
from llm_bridge.logic.chat_generate.model_message_converter.gemini_message_converter import convert_message_to_gemini
from llm_bridge.logic.chat_generate.model_message_converter.openai_message_converter import convert_message_to_openai
from llm_bridge.logic.chat_generate.model_message_converter.openai_responses_message_converter import \
    convert_message_to_openai_responses
from llm_bridge.type.message import Message
from llm_bridge.type.model_message.claude_message import ClaudeMessage
from llm_bridge.type.model_message.gemini_message import GeminiMessage
from llm_bridge.type.model_message.openai_message import OpenAIMessage
from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage


async def convert_messages_to_openai(messages: list[Message]) -> list[OpenAIMessage]:
    return [await convert_message_to_openai(message) for message in messages]


async def convert_messages_to_openai_responses(messages: list[Message]) -> list[OpenAIResponsesMessage]:
    return [await convert_message_to_openai_responses(message) for message in messages]


async def convert_messages_to_gemini(messages: list[Message]) -> list[GeminiMessage]:
    return [await convert_message_to_gemini(message) for message in messages]


async def convert_messages_to_claude(messages: list[Message]) -> list[ClaudeMessage]:
    return [await convert_message_to_claude(message) for message in messages]
