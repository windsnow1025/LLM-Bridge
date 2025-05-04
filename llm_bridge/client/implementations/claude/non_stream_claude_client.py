import logging
import re

import httpx
from fastapi import HTTPException

from llm_bridge.client.implementations.claude.claude_token_counter import count_claude_input_tokens, \
    count_claude_output_tokens
from llm_bridge.client.model_client.claude_client import ClaudeClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


class NonStreamClaudeClient(ClaudeClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            input_tokens = await count_claude_input_tokens(
                client=self.client,
                model=self.model,
                system=self.system,
                messages=self.messages
            )

            message = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=self.system,
                messages=serialize(self.messages)
            )

            content = message.content[0].text
            chat_response = ChatResponse(text=content)
            output_tokens = await count_claude_output_tokens(
                client=self.client,
                model=self.model,
                chat_response=chat_response,
            )
            return ChatResponse(
                text=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
