import anthropic
from anthropic.types import ThinkingConfigEnabledParam, AnthropicBetaParam
from anthropic.types.beta import BetaWebSearchTool20250305Param, BetaToolUnionParam, BetaCodeExecutionTool20250825Param

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

    max_tokens = min(
        32_000,  # Max output: Claude 4.5 64K; Claude 4.1 32K
        200_000 - input_tokens  # Context window: Claude Sonnet 4.5 beta: 1M; otherwise 200K
    )
    thinking = ThinkingConfigEnabledParam(
        type="enabled",
        budget_tokens=min(32_000, max_tokens) // 2
    )
    temperature = 1
    betas: list[AnthropicBetaParam] = [
        "context-1m-2025-08-07",
        "output-128k-2025-02-19",
        "code-execution-2025-08-25",
    ]
    tools: list[BetaToolUnionParam] = [
        BetaWebSearchTool20250305Param(
            type="web_search_20250305",
            name="web_search",
        ),
        BetaCodeExecutionTool20250825Param(
            type="code_execution_20250825",
            name="code_execution",
        )
    ]

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
