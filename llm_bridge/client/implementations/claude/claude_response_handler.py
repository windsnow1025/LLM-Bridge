from anthropic import AsyncAnthropic
from anthropic._response import AsyncBinaryAPIResponse
from anthropic.types.beta import BetaRawMessageStreamEvent, BetaThinkingDelta, BetaTextDelta, \
    BetaInputJSONDelta, BetaBashCodeExecutionToolResultBlock, \
    BetaTextEditorCodeExecutionToolResultBlock, BetaTextEditorCodeExecutionViewResultBlock, \
    BetaTextEditorCodeExecutionStrReplaceResultBlock, \
    BetaServerToolUseBlock, BetaBashCodeExecutionResultBlock, BetaTextBlock, BetaThinkingBlock, \
    BetaMessage, FileMetadata, BetaRawMessageStartEvent, BetaRawContentBlockStartEvent, \
    BetaRawContentBlockDeltaEvent, BetaRawMessageDeltaEvent
from anthropic.types.beta.beta_raw_content_block_start_event import ContentBlock

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

    if isinstance(content_block, BetaTextBlock):
        text += content_block.text

    elif isinstance(content_block, BetaThinkingBlock):
        thought += content_block.thinking

    elif isinstance(content_block, BetaServerToolUseBlock):
        code += str(content_block.input)

    elif isinstance(content_block, BetaBashCodeExecutionToolResultBlock):
        result = content_block.content
        if isinstance(result, BetaBashCodeExecutionResultBlock):
            code_output += result.stdout
            file_ids = [output.file_id for output in result.content]
            for file_id in file_ids:
                file = await download_claude_file(client, file_id)
                files.append(file)

    elif isinstance(content_block, BetaTextEditorCodeExecutionToolResultBlock):
        result = content_block.content
        if isinstance(result, BetaTextEditorCodeExecutionViewResultBlock):
            code_output += result.content
        elif isinstance(result, BetaTextEditorCodeExecutionStrReplaceResultBlock):
            code_output += result.lines

    return ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
    )


async def process_claude_non_stream_response(
        message: BetaMessage,
        client: AsyncAnthropic,
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

    return ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


class ClaudeResponseHandler:
    def __init__(self):
        self.prev_cumulative_output_tokens: int = 0

    async def process_claude_stream_response(
            self,
            event: BetaRawMessageStreamEvent,
            client: AsyncAnthropic,
    ) -> ChatResponse:
        text = ""
        thought = ""
        code = ""
        code_output = ""
        files: list[File] = []
        input_tokens: int | None = None
        output_tokens: int = 0

        if isinstance(event, BetaRawMessageStartEvent):
            input_tokens = event.message.usage.input_tokens

        elif isinstance(event, BetaRawContentBlockDeltaEvent):
            event_delta = event.delta

            if isinstance(event_delta, BetaTextDelta):
                text += event_delta.text

            elif isinstance(event_delta, BetaThinkingDelta):
                thought += event_delta.thinking

            elif isinstance(event_delta, BetaInputJSONDelta):
                code += event_delta.partial_json

        elif isinstance(event, BetaRawContentBlockStartEvent):
            content_block_chat_response = await process_content_block(event.content_block, client)
            text += content_block_chat_response.text
            thought += content_block_chat_response.thought
            code += content_block_chat_response.code
            code_output += content_block_chat_response.code_output
            if file := content_block_chat_response.files:
                files.extend(file)

        elif isinstance(event, BetaRawMessageDeltaEvent):
            cumulative_output_tokens = event.usage.output_tokens
            output_tokens = cumulative_output_tokens - self.prev_cumulative_output_tokens
            self.prev_cumulative_output_tokens = cumulative_output_tokens

        return ChatResponse(
            text=text,
            thought=thought,
            code=code,
            code_output=code_output,
            files=files,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
