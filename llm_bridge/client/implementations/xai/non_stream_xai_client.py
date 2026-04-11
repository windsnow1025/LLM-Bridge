import logging
import re

from fastapi import HTTPException
from xai_sdk.aio.chat import Chat
from xai_sdk.chat import Response

from llm_bridge.client.model_client.xai_client import XAIClient
from llm_bridge.type.chat_response import ChatResponse


class NonStreamXAIClient(XAIClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            chat: Chat = self.client.chat.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                store_messages=False,
                tools=self.tools,
            )

            response: Response = await chat.sample()

            text = response.content
            thought = response.reasoning_content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return ChatResponse(
                text=text,
                thought=thought,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:
            logging.exception(e)
            match = re.search(r'\d{3}', str(e))
            if match:
                error_code = int(match.group(0))
            else:
                error_code = 500

            raise HTTPException(status_code=error_code, detail=str(e))
