import logging
import re
from typing import AsyncGenerator

import httpx
from fastapi import HTTPException

from llm_bridge.client.model_client.claude_client import ClaudeClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


class StreamClaudeClient(ClaudeClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            try:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    temperature=self.temperature,
                    system=self.system,
                    messages=serialize(self.messages)
                ) as stream:
                    async for response_delta in stream.text_stream:
                        yield ChatResponse(text=response_delta)
            except Exception as e:
                logging.exception(e)
                yield ChatResponse(error=repr(e))

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
