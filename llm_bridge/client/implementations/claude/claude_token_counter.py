import anthropic
from llm_bridge import serialize, ChatResponse, Role

from llm_bridge.type.model_message.claude_message import ClaudeMessage, TextContent


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
    messages = [
        ClaudeMessage(role=Role.Assistant, content=[
            TextContent(type="text", text=chat_response.text),
        ]),
    ]

    response = await client.messages.count_tokens(
        model=model,
        messages=serialize(messages),
    )

    return response.input_tokens
