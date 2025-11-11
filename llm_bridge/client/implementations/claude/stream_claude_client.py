import logging
import re
from typing import AsyncGenerator

import httpx
from fastapi import HTTPException

from llm_bridge.client.implementations.claude.claude_response_handler import process_claude_stream_response
from llm_bridge.client.model_client.claude_client import ClaudeClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize

class StreamClaudeClient(ClaudeClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            try:
                async with self.client.beta.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system,
                    messages=serialize(self.messages),
                    thinking=self.thinking,
                    betas=self.betas,
                    tools=self.tools,
                ) as stream:
                    async for event in stream:
                        yield await process_claude_stream_response(
                            event=event,
                            input_tokens=self.input_tokens,
                            client=self.client,
                            model=self.model,
                        )

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
