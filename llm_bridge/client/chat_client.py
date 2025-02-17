from typing import AsyncGenerator

from llm_bridge.type.chat_response import ChatResponse


class ChatClient:
    async def generate_response(self) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
