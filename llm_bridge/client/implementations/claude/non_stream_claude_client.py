import logging
import re

import httpx
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage
from fastapi import HTTPException

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_output_tokens
from llm_bridge.client.model_client.claude_client import ClaudeClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


async def process_claude_non_stream_response(
        message: BetaMessage,
        input_tokens: int,
        client: AsyncAnthropic,
        model: str,
) -> ChatResponse:
    text = ""

    for content in message.content:
        if content.type == "thinking":
            text += "# Model Thought:\n\n"
            text += content.thinking
        if content.type == "text":
            text += "\n\n# Model Response:\n\n"
            text += content.text
            # Unable to test since streaming Claude is currently not allowed
            if citations := content.citations:
                for citation in citations:
                    text += f"([{citation.title}]({citation.url})) "

    chat_response = ChatResponse(text=text)
    output_tokens = await count_claude_output_tokens(
        client=client,
        model=model,
        chat_response=chat_response,
    )
    return ChatResponse(
        text=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class NonStreamClaudeClient(ClaudeClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            message = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system,
                messages=serialize(self.messages),
                thinking=self.thinking,
                betas=self.betas,
                tools=self.tools,
            )

            return await process_claude_non_stream_response(
                message=message,
                input_tokens=self.input_tokens,
                client=self.client,
                model=self.model,
            )
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            text = e.response.text
            raise HTTPException(status_code=status_code, detail=text)
        except Exception as e:
            logging.exception(e)
            match = re.search(r'\d{3}', str(e))
            if match:
                error_code = int(match.group(0))
            else:
                error_code = 500

            raise HTTPException(status_code=error_code, detail=str(e))
