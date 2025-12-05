from typing import AsyncGenerator, Any

from llm_bridge import *


async def workflow(
        api_keys: dict[str, str],
        messages: list[Message],
        model: str,
        api_type: str,
        temperature: float,
        stream: bool,
        thought: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None,
) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
    await preprocess_messages(messages, api_type)

    chat_client = await create_chat_client(
        api_keys=api_keys,
        messages=messages,
        model=model,
        api_type=api_type,
        temperature=temperature,
        stream=stream,
        thought=thought,
        code_execution=code_execution,
        structured_output_schema=structured_output_schema,
    )

    if stream:
        return chat_client.generate_stream_response()
    else:
        return await chat_client.generate_non_stream_response()
