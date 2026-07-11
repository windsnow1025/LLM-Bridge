import logging
import re
from typing import AsyncGenerator

from fastapi import HTTPException
from xai_sdk.aio.chat import Chat

from llm_bridge.client.model_client.xai_client import XAIClient
from llm_bridge.type.chat_response import ChatResponse


class StreamXAIClient(XAIClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            chat: Chat = self.client.chat.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                store_messages=False,
                tools=self.tools,
                reasoning_effort=self.reasoning_effort,
                response_format=self.response_format,
            )
        except Exception as e:
            logging.exception(e)
            match = re.search(r'\d{3}', str(e))
            if match:
                error_code = int(match.group(0))
            else:
                error_code = 500

            raise HTTPException(status_code=error_code, detail=str(e))

        try:
            prev_cumulative_output_tokens: int = 0
            async for response, chunk in chat.stream():
                cumulative_output_tokens = chunk.proto.usage.completion_tokens
                output_tokens = cumulative_output_tokens - prev_cumulative_output_tokens
                prev_cumulative_output_tokens = cumulative_output_tokens

                yield ChatResponse(
                    text=chunk.content,
                    thought=chunk.reasoning_content,
                    input_tokens=chunk.proto.usage.prompt_tokens,
                    output_tokens=output_tokens,
                )
        except Exception as e:
            logging.exception(e)
            yield ChatResponse(error=repr(e))
