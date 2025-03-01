from llm_bridge import *


async def workflow(
        api_keys: dict[str, str],
        messages: list[Message],
        model: str,
        api_type: str,
        temperature: float,
        stream: bool
):
    await preprocess_messages(messages)

    chat_client = await create_chat_client(
        messages=messages,
        model=model,
        api_type=api_type,
        temperature=temperature,
        stream=stream,
        api_keys=api_keys,
    )

    return await chat_client.generate_response()


