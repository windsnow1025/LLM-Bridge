from enum import Enum
from pprint import pprint

from anthropic import BetaMessageStreamEvent, AsyncAnthropic

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_output_tokens
from llm_bridge.type.chat_response import ChatResponse


class PrintingStatus(Enum):
    Start = "start"
    Thought = "thought"
    Response = "response"


class ClaudeStreamResponseHandler:
    def __init__(self):
        self.printing_status = PrintingStatus.Start

    async def process_claude_stream_response(
            self,
            event: BetaMessageStreamEvent,
            input_tokens: int,
            client: AsyncAnthropic,
            model: str,
    ) -> ChatResponse:
        text = ""
        if event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                if self.printing_status == PrintingStatus.Start:
                    text += "# Model Thought:\n\n"
                    self.printing_status = PrintingStatus.Thought
                text += event.delta.thinking
            elif event.delta.type == "text_delta":
                if self.printing_status == PrintingStatus.Thought:
                    text += "\n\n# Model Response:\n\n"
                    self.printing_status = PrintingStatus.Response
                text += event.delta.text
        elif event.type == "citation":
            citation = event.citation
            text += f"([{citation.title}]({citation.url})) "
        chat_response = ChatResponse(text=text)
        output_tokens = await count_claude_output_tokens(
            client=client,
            model=model,
            chat_response=chat_response,
        )
        return ChatResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
