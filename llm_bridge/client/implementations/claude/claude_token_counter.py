import anthropic

from llm_bridge.type.model_message.claude_message import ClaudeMessage
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
