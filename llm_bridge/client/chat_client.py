from typing import AsyncGenerator

from llm_bridge.type.chat_response import ChatResponse


class ChatClient:
    async def generate_non_stream_response(self) -> ChatResponse:
        raise NotImplementedError
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError
