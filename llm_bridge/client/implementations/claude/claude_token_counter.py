from pprint import pprint

import anthropic
from anthropic.types import TextBlockParam

from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.model_message.claude_message import ClaudeMessage, ClaudeRole
from llm_bridge.type.serializer import serialize


async def count_claude_input_tokens(
        client: anthropic.AsyncAnthropic,
        model: str,
        system: str,
        messages: list[ClaudeMessage],
) -> int:
    response = await client.messages.count_tokens(
        model=model,
        system=system,
        messages=serialize(messages),
    )

    return response.input_tokens


async def count_claude_output_tokens(
        client: anthropic.AsyncAnthropic,
        model: str,
        chat_response: ChatResponse,
) -> int:
    text = ""
    text += chat_response.text.strip()
    text += chat_response.thought.strip()
    text += chat_response.code.strip()
    text += chat_response.code_output.strip()
    if text == "":
        return 0

    messages = [
        ClaudeMessage(
            role=ClaudeRole.Assistant,
            content=[
                TextBlockParam(type="text", text=text),
            ]
        ),
    ]

    response = await client.messages.count_tokens(
        model=model,
        messages=serialize(messages),
    )

    return response.input_tokens
