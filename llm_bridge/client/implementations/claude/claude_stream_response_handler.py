from anthropic import BetaMessageStreamEvent, AsyncAnthropic

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

        if event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                thought += event.delta.thinking
            elif event.delta.type == "text_delta":
                text += event.delta.text

        chat_response = ChatResponse(
            text=text,
            thought=thought,
        )
        output_tokens = await count_claude_output_tokens(
            client=client,
            model=model,
            chat_response=chat_response,
        )
        return ChatResponse(
            text=text,
            thought=thought,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
