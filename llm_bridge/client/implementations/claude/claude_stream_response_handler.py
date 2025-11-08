from anthropic import BetaMessageStreamEvent, AsyncAnthropic
from anthropic.types.beta import BetaRawContentBlockDelta, BetaThinkingDelta, BetaTextDelta, BetaInputJSONDelta, \
    BetaBashCodeExecutionToolResultBlock, \
    BetaTextEditorCodeExecutionToolResultBlock
from anthropic.types.beta.beta_raw_content_block_start_event import ContentBlock

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_output_tokens
from llm_bridge.type.chat_response import ChatResponse


class ClaudeStreamResponseHandler:
    async def process_claude_stream_response(
            self,
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
            event_content_block: ContentBlock = event.content_block

            if event_content_block.type == "bash_code_execution_tool_result":
                bash_code_execution_tool_result_block: BetaBashCodeExecutionToolResultBlock = event_content_block
                if bash_code_execution_tool_result_block.content.type == "bash_code_execution_result":
                    code_output += event_content_block.content.stdout

            elif event_content_block.type == "text_editor_code_execution_tool_result":
                text_editor_code_execution_tool_result: BetaTextEditorCodeExecutionToolResultBlock = event_content_block
                if text_editor_code_execution_tool_result.content.type == "text_editor_code_execution_view_result":
                    code_output += event_content_block.content.content

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
