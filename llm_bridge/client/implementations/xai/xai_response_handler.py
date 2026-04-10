from xai_sdk.chat import Response, Chunk

from llm_bridge.type.chat_response import ChatResponse


async def process_xai_non_stream_response(response: Response) -> ChatResponse:
    pass


def process_xai_stream_chunk(chunk: Chunk) -> ChatResponse:
    pass
