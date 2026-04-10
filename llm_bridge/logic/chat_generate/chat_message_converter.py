from llm_bridge.logic.chat_generate.model_message_converter.claude_message_converter import convert_message_to_claude
from llm_bridge.logic.chat_generate.model_message_converter.gemini_message_converter import convert_message_to_gemini
from llm_bridge.logic.chat_generate.model_message_converter.openai_completion_message_converter import convert_message_to_openai_completion
from llm_bridge.logic.chat_generate.model_message_converter.openai_responses_message_converter import \
    convert_message_to_openai_responses
from llm_bridge.logic.chat_generate.model_message_converter.xai_message_converter import convert_message_to_xai
from llm_bridge.type.message import Message
from llm_bridge.type.model_message.claude_message import ClaudeMessage
from llm_bridge.type.model_message.gemini_message import GeminiMessage
from llm_bridge.type.model_message.openai_completion_message import OpenAICompletionMessage
from llm_bridge.type.model_message.openai_responses_message import OpenAIResponsesMessage
from llm_bridge.type.model_message.xai_message import XAIMessage


async def convert_messages_to_openai(messages: list[Message]) -> list[OpenAICompletionMessage]:
    return [await convert_message_to_openai_completion(message) for message in messages]


async def convert_messages_to_openai_responses(messages: list[Message]) -> list[OpenAIResponsesMessage]:
    return [await convert_message_to_openai_responses(message) for message in messages]


async def convert_messages_to_gemini(messages: list[Message]) -> list[GeminiMessage]:
    return [await convert_message_to_gemini(message) for message in messages]


async def convert_messages_to_claude(messages: list[Message]) -> list[ClaudeMessage]:
    return [await convert_message_to_claude(message) for message in messages]


async def convert_messages_to_xai(messages: list[Message]) -> list[XAIMessage]:
    return [await convert_message_to_xai(message) for message in messages]
