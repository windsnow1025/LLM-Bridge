from anthropic import BetaMessageStreamEvent, AsyncAnthropic
from anthropic.types.beta import BetaRawContentBlockDelta, BetaThinkingDelta, BetaTextDelta, BetaInputJSONDelta, \
    BetaBashCodeExecutionToolResultBlock, \
    BetaTextEditorCodeExecutionToolResultBlock, BetaTextEditorCodeExecutionViewResultBlock, \
    BetaTextEditorCodeExecutionStrReplaceResultBlock, \
    BetaServerToolUseBlock, BetaBashCodeExecutionResultBlock, BetaTextBlock, BetaThinkingBlock
from anthropic.types.beta.beta_raw_content_block_start_event import ContentBlock

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_output_tokens
from llm_bridge.type.chat_response import ChatResponse


def process_content_block(content_block: ContentBlock) -> ChatResponse:
    text = ""
    thought = ""
    code = ""
    code_output = ""

    if content_block.type == "text":
        text_block: BetaTextBlock = content_block
        text += text_block.text

    elif content_block.type == "thinking":
        thinking_block: BetaThinkingBlock = content_block
        thought += thinking_block.thinking

    elif content_block.type == "server_tool_use":
        server_tool_use_block: BetaServerToolUseBlock = content_block
        code += str(server_tool_use_block.input)

    elif content_block.type == "bash_code_execution_tool_result":
        bash_code_execution_tool_result_block: BetaBashCodeExecutionToolResultBlock = content_block
        if bash_code_execution_tool_result_block.content.type == "bash_code_execution_result":
            content: BetaBashCodeExecutionResultBlock = content_block.content
            code_output += content.stdout

    elif content_block.type == "text_editor_code_execution_tool_result":
        text_editor_code_execution_tool_result: BetaTextEditorCodeExecutionToolResultBlock = content_block
        if text_editor_code_execution_tool_result.content.type == "text_editor_code_execution_view_result":
            content: BetaTextEditorCodeExecutionViewResultBlock = content_block.content
            code_output += content.content
        elif text_editor_code_execution_tool_result.content.type == "text_editor_code_execution_str_replace_result":
            content: BetaTextEditorCodeExecutionStrReplaceResultBlock = content_block.content
            code_output += content.lines

    return ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
    )


async def process_claude_stream_response(
        event: BetaMessageStreamEvent,
        input_tokens: int,
        client: AsyncAnthropic,
        model: str,
) -> ChatResponse:
    text = ""
    thought = ""
    code = ""
    code_output = ""

    if event.type == "content_block_delta":
        event_delta: BetaRawContentBlockDelta = event.delta

        if event_delta.type == "text_delta":
            text_delta: BetaTextDelta = event_delta
            text += text_delta.text

        elif event_delta.type == "thinking_delta":
            thinking_delta: BetaThinkingDelta = event_delta
            thought += thinking_delta.thinking

        elif event_delta.type == "input_json_delta":
            input_json_delta: BetaInputJSONDelta = event_delta
            code += input_json_delta.partial_json

    if event.type == "content_block_start":
        content_block: ContentBlock = event.content_block
        content_block_chat_response = process_content_block(content_block)
        text += content_block_chat_response.text
        thought += content_block_chat_response.thought
        code += content_block_chat_response.code
        code_output += content_block_chat_response.code_output

    chat_response = ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
    )
    output_tokens = await count_claude_output_tokens(
        client=client,
        model=model,
        chat_response=chat_response,
    )
    return ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
