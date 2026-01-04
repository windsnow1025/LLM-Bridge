import logging
import re

import httpx
from fastapi import HTTPException

from llm_bridge.client.implementations.claude.claude_response_handler import process_claude_non_stream_response
from llm_bridge.client.model_client.claude_client import ClaudeClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


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
                betas=self.betas,
                tools=self.tools,
                thinking=self.thinking,
                output_format=self.output_format,
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
