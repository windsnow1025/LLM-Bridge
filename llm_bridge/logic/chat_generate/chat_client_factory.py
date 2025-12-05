from typing import Any

from fastapi import HTTPException

from llm_bridge.client.chat_client import ChatClient
from llm_bridge.logic.chat_generate.model_client_factory.claude_client_factory import create_claude_client
from llm_bridge.logic.chat_generate.model_client_factory.gemini_client_factory import create_gemini_client
from llm_bridge.logic.chat_generate.model_client_factory.openai_client_factory import create_openai_client
from llm_bridge.type.message import Message


async def create_chat_client(
        api_keys: dict,
        messages: list[Message],
        model: str,
        api_type: str,
        temperature: float,
        stream: bool,
        thought: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None,
) -> ChatClient:
    if api_type == 'OpenAI':
        return await create_openai_client(
            api_keys={"OPENAI_API_KEY": api_keys["OPENAI_API_KEY"]},
            messages=messages,
            model=model,
            api_type=api_type,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            structured_output_schema=structured_output_schema,
        )
    elif api_type == 'OpenAI-Azure':
        return await create_openai_client(
            api_keys={
                "AZURE_API_KEY": api_keys["AZURE_API_KEY"],
                "AZURE_API_BASE": api_keys["AZURE_API_BASE"]
            },
            messages=messages,
            model=model,
            api_type=api_type,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            structured_output_schema=structured_output_schema,
        )
    elif api_type == 'OpenAI-GitHub':
        return await create_openai_client(
            api_keys={"GITHUB_API_KEY": api_keys["GITHUB_API_KEY"]},
            messages=messages,
            model=model,
            api_type=api_type,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            structured_output_schema=structured_output_schema,
        )
    elif api_type == 'Grok':
        return await create_openai_client(
            api_keys={"XAI_API_KEY": api_keys["XAI_API_KEY"]},
            messages=messages,
            model=model,
            api_type=api_type,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            structured_output_schema=structured_output_schema,
        )
    elif api_type == 'Gemini-Free':
        return await create_gemini_client(
            api_key=api_keys["GEMINI_FREE_API_KEY"],
            vertexai=False,
            messages=messages,
            model=model,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            structured_output_schema=structured_output_schema,
        )
    elif api_type == 'Gemini-Paid':
        return await create_gemini_client(
            api_key=api_keys["GEMINI_PAID_API_KEY"],
            vertexai=False,
            messages=messages,
            model=model,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            structured_output_schema=structured_output_schema,
        )
    elif api_type == 'Gemini-Vertex':
        return await create_gemini_client(
            api_key=api_keys["GEMINI_VERTEX_API_KEY"],
            messages=messages,
            model=model,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            vertexai=True,
            structured_output_schema=structured_output_schema,
        )
    elif api_type == 'Claude':
        return await create_claude_client(
            api_key=api_keys["ANTHROPIC_API_KEY"],
            messages=messages,
            model=model,
            temperature=temperature,
            stream=stream,
            thought=thought,
            code_execution=code_execution,
            structured_output_schema=structured_output_schema,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid API type")
