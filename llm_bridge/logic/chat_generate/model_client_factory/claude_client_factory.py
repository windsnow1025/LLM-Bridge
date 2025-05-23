import anthropic
from anthropic.types import ThinkingConfigEnabledParam

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_input_tokens
from llm_bridge.client.implementations.claude.non_stream_claude_client import NonStreamClaudeClient
from llm_bridge.client.implementations.claude.stream_claude_client import StreamClaudeClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_claude
from llm_bridge.logic.message_preprocess.message_preprocessor import extract_system_messages
from llm_bridge.type.message import Message


async def create_claude_client(
        messages: list[Message],
        model: str,
        temperature: float,
        stream: bool,
        api_key: str,
):
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
    )

    system = extract_system_messages(messages)
    if system == "":
        system = "/"

    claude_messages = await convert_messages_to_claude(messages)

    input_tokens = await count_claude_input_tokens(
        client=client,
        model=model,
        system=system,
        messages=claude_messages,
    )

    max_tokens = min(32000, 200000 - input_tokens)
    thinking = ThinkingConfigEnabledParam(
        type="enabled",
        budget_tokens=16000
    )
    temperature = 1
    betas = ["output-128k-2025-02-19"]
    tools = [{
        "type": "web_search_20250305",
        "name": "web_search",
    }]

    if stream:
        return StreamClaudeClient(
            model=model,
            messages=claude_messages,
            temperature=temperature,
            system=system,
            client=client,
            max_tokens=max_tokens,
            thinking=thinking,
            betas=betas,
            input_tokens=input_tokens,
            tools=tools,
        )
    else:
        return NonStreamClaudeClient(
            model=model,
            messages=claude_messages,
            temperature=temperature,
            system=system,
            client=client,
            max_tokens=max_tokens,
            thinking=thinking,
            betas=betas,
            input_tokens=input_tokens,
            tools=tools,
        )


