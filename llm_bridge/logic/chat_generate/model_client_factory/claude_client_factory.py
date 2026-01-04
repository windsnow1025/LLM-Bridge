from typing import Any

import anthropic
from anthropic import Omit, transform_schema
from anthropic.types import ThinkingConfigEnabledParam, AnthropicBetaParam
from anthropic.types.beta import BetaWebSearchTool20250305Param, BetaToolUnionParam, BetaCodeExecutionTool20250825Param, \
    BetaToolBash20250124Param

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_input_tokens
from llm_bridge.client.implementations.claude.non_stream_claude_client import NonStreamClaudeClient
from llm_bridge.client.implementations.claude.stream_claude_client import StreamClaudeClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_claude
from llm_bridge.logic.message_preprocess.message_preprocessor import extract_system_messages
from llm_bridge.type.message import Message


async def create_claude_client(
        api_key: str,
        messages: list[Message],
        model: str,
        temperature: float,
        stream: bool,
        thought: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None
):
    omit = Omit()

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

    context_window = 200_000
    if model in ["claude-sonnet-4-5"]:
        context_window = 1_000_000
    max_output = 64_000
    max_tokens = min(
        max_output,
        context_window - input_tokens,
    )
    thinking = omit
    if thought:
        thinking = ThinkingConfigEnabledParam(
            type="enabled",
            budget_tokens=max(1024, max_tokens // 2),  # Minimum budget tokens: 1024
        )
        temperature = 1
    betas: list[AnthropicBetaParam] = [
        "context-1m-2025-08-07",
        "output-128k-2025-02-19",
        "code-execution-2025-08-25",
        "files-api-2025-04-14",
        "structured-outputs-2025-11-13",
    ]
    tools: list[BetaToolUnionParam] = []
    tools.append(
        BetaWebSearchTool20250305Param(
            type="web_search_20250305",
            name="web_search",
        )
    )
    if code_execution:
        tools.append(
            BetaToolBash20250124Param(
                type="bash_20250124",
                name="bash",
            )
        )
        tools.append(
            BetaCodeExecutionTool20250825Param(
                type="code_execution_20250825",
                name="code_execution",
            )
        )

    output_format = omit
    # if structured_output_schema: # Claude output format raises: TypeError: unhashable type: 'dict'
    #     output_format = {
    #         "type": "json_schema",
    #         "schema": transform_schema(structured_output_schema),
    #     }

    if stream:
        return StreamClaudeClient(
            model=model,
            messages=claude_messages,
            temperature=temperature,
            system=system,
            client=client,
            max_tokens=max_tokens,
            betas=betas,
            input_tokens=input_tokens,
            tools=tools,
            thinking=thinking,
            output_format=output_format,
        )
    else:
        return NonStreamClaudeClient(
            model=model,
            messages=claude_messages,
            temperature=temperature,
            system=system,
            client=client,
            max_tokens=max_tokens,
            betas=betas,
            input_tokens=input_tokens,
            tools=tools,
            thinking=thinking,
            output_format=output_format,
        )
