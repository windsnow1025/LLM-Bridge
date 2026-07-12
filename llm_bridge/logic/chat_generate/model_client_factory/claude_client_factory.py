from typing import Any

import anthropic
from anthropic import Omit, transform_schema
from anthropic.types import AnthropicBetaParam
from anthropic.types.beta import BetaWebSearchTool20250305Param, BetaToolUnionParam, \
    BetaJSONOutputFormatParam, BetaOutputConfigParam, BetaThinkingConfigParam, BetaThinkingConfigEnabledParam, \
    BetaThinkingConfigAdaptiveParam, BetaWebFetchTool20260209Param, BetaCitationsConfigParam, \
    BetaCacheControlEphemeralParam

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_input_tokens
from llm_bridge.client.implementations.claude.non_stream_claude_client import NonStreamClaudeClient
from llm_bridge.client.implementations.claude.stream_claude_client import StreamClaudeClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_claude
from llm_bridge.logic.message_process.message_processor import extract_system_messages
from llm_bridge.type.message import Message


async def create_claude_client(
        api_key: str,
        messages: list[Message],
        model: str,
        temperature: float,
        stream: bool,
        thought: bool,
        web_search: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None
) -> StreamClaudeClient | NonStreamClaudeClient:
    omit = Omit()

    client = anthropic.AsyncAnthropic(
        api_key=api_key,
    )

    system = await extract_system_messages(messages)
    if system == "":
        system = "/"

    claude_messages = await convert_messages_to_claude(messages)

    input_tokens = await count_claude_input_tokens(
        client=client,
        model=model,
        system=system,
        messages=claude_messages,
    )

    context_window = 1_000_000
    max_output = 64_000
    max_tokens = min(
        max_output,
        context_window - input_tokens,
    )

    cache_control = BetaCacheControlEphemeralParam(type="ephemeral")

    thinking: BetaThinkingConfigParam | Omit = omit
    if thought:
        thinking = BetaThinkingConfigAdaptiveParam(type="adaptive", display="summarized")

    betas: list[AnthropicBetaParam] | Omit = omit

    tools: list[BetaToolUnionParam] = []
    if web_search:
        tools.append(
            BetaWebSearchTool20250305Param(
                type="web_search_20250305",
                name="web_search",
            )
        )
    if code_execution:
        tools.append(
            # Code Execution auto-injected
            BetaWebFetchTool20260209Param(
                type="web_fetch_20260209",
                name="web_fetch",
                citations=BetaCitationsConfigParam(enabled=True)
            )
        )

    output_config: BetaOutputConfigParam | Omit = omit
    if thought or structured_output_schema:
        output_config: BetaOutputConfigParam = BetaOutputConfigParam()
        if thought:
            output_config["effort"] = "high"
        if structured_output_schema:
            output_config["format"] = BetaJSONOutputFormatParam(
                type="json_schema",
                schema=transform_schema(structured_output_schema),
            )

    if stream:
        return StreamClaudeClient(
            model=model,
            messages=claude_messages,
            system=system,
            max_tokens=max_tokens,
            client=client,
            betas=betas,
            tools=tools,
            cache_control=cache_control,
            thinking=thinking,
            output_config=output_config,
        )
    else:
        return NonStreamClaudeClient(
            model=model,
            messages=claude_messages,
            system=system,
            max_tokens=max_tokens,
            client=client,
            betas=betas,
            tools=tools,
            cache_control=cache_control,
            thinking=thinking,
            output_config=output_config,
        )
