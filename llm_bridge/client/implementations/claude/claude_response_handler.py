from anthropic import BetaMessageStreamEvent, AsyncAnthropic
from anthropic._response import AsyncBinaryAPIResponse
from anthropic.types.beta import BetaRawContentBlockDelta, BetaThinkingDelta, BetaTextDelta, BetaInputJSONDelta, \
    BetaBashCodeExecutionToolResultBlock, \
    BetaTextEditorCodeExecutionToolResultBlock, BetaTextEditorCodeExecutionViewResultBlock, \
    BetaTextEditorCodeExecutionStrReplaceResultBlock, \
    BetaServerToolUseBlock, BetaBashCodeExecutionResultBlock, BetaTextBlock, BetaThinkingBlock, \
    BetaBashCodeExecutionOutputBlock, BetaMessage, FileMetadata
from anthropic.types.beta.beta_raw_content_block_start_event import ContentBlock

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_output_tokens
from llm_bridge.logic.chat_generate.media_processor import bytes_to_base64
from llm_bridge.type.chat_response import ChatResponse, File


async def download_claude_file(client: AsyncAnthropic, file_id: str) -> File:
    file_metadata: FileMetadata = await client.beta.files.retrieve_metadata(file_id)
    file_content: AsyncBinaryAPIResponse = await client.beta.files.download(file_id)
    data = await file_content.read()
    return File(
        name=file_metadata.filename,
        data=bytes_to_base64(data),
        type=file_metadata.mime_type,
    )

async def process_content_block(
        content_block: ContentBlock, client: AsyncAnthropic
) -> ChatResponse:
    text: str = ""
    thought: str = ""
    code: str = ""
    code_output: str = ""
    files: list[File] = []

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
            result: BetaBashCodeExecutionResultBlock = content_block.content
            code_output += result.stdout
            outputs: list[BetaBashCodeExecutionOutputBlock] = result.content
            file_ids = [output.file_id for output in outputs]
            for file_id in file_ids:
                file = await download_claude_file(client, file_id)
                files.append(file)

    elif content_block.type == "text_editor_code_execution_tool_result":
        text_editor_code_execution_tool_result: BetaTextEditorCodeExecutionToolResultBlock = content_block
        if text_editor_code_execution_tool_result.content.type == "text_editor_code_execution_view_result":
            result: BetaTextEditorCodeExecutionViewResultBlock = content_block.content
            code_output += result.content
        elif text_editor_code_execution_tool_result.content.type == "text_editor_code_execution_str_replace_result":
            result: BetaTextEditorCodeExecutionStrReplaceResultBlock = content_block.content
            code_output += result.lines

    return ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
    )


async def build_chat_response_with_tokens(
        text: str,
        thought: str,
        code: str,
        code_output: str,
        files: list[File],
        input_tokens: int,
        client: AsyncAnthropic,
        model: str,
) -> ChatResponse:
    chat_response = ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
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
        files=files,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def process_claude_non_stream_response(
        message: BetaMessage,
        input_tokens: int,
        client: AsyncAnthropic,
        model: str,
) -> ChatResponse:
    text = ""
    thought = ""
    code = ""
    code_output = ""
    files: list[File] = []

    for content_block in message.content:
        content_block_chat_response = await process_content_block(content_block, client)
        text += content_block_chat_response.text
        thought += content_block_chat_response.thought
        code += content_block_chat_response.code
        code_output += content_block_chat_response.code_output
        files.extend(content_block_chat_response.files)

    return await build_chat_response_with_tokens(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
        input_tokens=input_tokens,
        client=client,
        model=model,
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
    files: list[File] = []

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
        content_block_chat_response = await process_content_block(content_block, client)
        text += content_block_chat_response.text
        thought += content_block_chat_response.thought
        code += content_block_chat_response.code
        code_output += content_block_chat_response.code_output
        files.extend(content_block_chat_response.files)

    return await build_chat_response_with_tokens(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
        input_tokens=input_tokens,
        client=client,
        model=model,
    )
