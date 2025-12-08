from typing import Any

from google import genai
from google.genai import types
from google.genai.types import Modality, MediaResolution

from llm_bridge.client.implementations.gemini.non_stream_gemini_client import NonStreamGeminiClient
from llm_bridge.client.implementations.gemini.stream_gemini_client import StreamGeminiClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_gemini
from llm_bridge.logic.message_preprocess.message_preprocessor import extract_system_messages
from llm_bridge.type.message import Message


async def create_gemini_client(
        api_key: str,
        vertexai: bool,
        messages: list[Message],
        model: str,
        temperature: float,
        stream: bool,
        thought: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None,
):
    client = genai.Client(
        vertexai=vertexai,
        api_key=api_key,
    )

    system_instruction = extract_system_messages(messages) or " "
    tools = []
    thinking_config = None
    response_modalities = [Modality.TEXT]

    tools.append(
        types.Tool(
            google_search=types.GoogleSearch()
        )
    )
    if thought:
        thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,
        )
    if "image" not in model:
        tools.append(
            types.Tool(
                url_context=types.UrlContext()
            )
        )
        if not vertexai:
            if code_execution:
                tools.append(
                    types.Tool(
                        code_execution=types.ToolCodeExecution()
                    )
                )
    if "image" in model:
        response_modalities = [Modality.TEXT, Modality.IMAGE]

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ],
        tools=tools,
        thinking_config=thinking_config,
        response_modalities=response_modalities,
    )

    if vertexai:
        config.media_resolution=MediaResolution.MEDIA_RESOLUTION_HIGH

    if structured_output_schema:
        config.response_mime_type = "application/json"
        config.response_json_schema = structured_output_schema

    gemini_messages = await convert_messages_to_gemini(messages)

    if stream:
        return StreamGeminiClient(
            model=model,
            messages=gemini_messages,
            temperature=temperature,
            client=client,
            config=config,
        )
    else:
        return NonStreamGeminiClient(
            model=model,
            messages=gemini_messages,
            temperature=temperature,
            client=client,
            config=config,
        )
