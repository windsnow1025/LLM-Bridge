from google import genai
from google.genai import types
from google.genai._api_client import HttpOptions
from google.genai.types import Modality

from llm_bridge.client.implementations.gemini.non_stream_gemini_client import NonStreamGeminiClient
from llm_bridge.client.implementations.gemini.stream_gemini_client import StreamGeminiClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_gemini
from llm_bridge.logic.message_preprocess.message_preprocessor import extract_system_messages
from llm_bridge.type.message import Message


async def create_gemini_client(
        messages: list[Message],
        model: str,
        temperature: float,
        stream: bool,
        api_key: str,
        vertexai: bool,
):
    client = genai.Client(
        vertexai=vertexai,
        api_key=api_key,
    )

    system_instruction = None
    tools = []
    thinking_config = None
    response_modalities = [Modality.TEXT]

    system_instruction = extract_system_messages(messages) or " "
    if "image" not in model and not vertexai:
        tools.append(
            types.Tool(
                google_search=types.GoogleSearch()
            )
        )
        tools.append(
            types.Tool(
                url_context=types.UrlContext()
            )
        )
        tools.append(
            types.Tool(
                code_execution=types.ToolCodeExecution()
            )
        )
    if "image" not in model and vertexai:
        tools.append(
            types.Tool(
                google_search=types.GoogleSearch()
            )
        )
        tools.append(
            types.Tool(
                url_context=types.UrlContext()
            )
        )
    if "image" not in model:
        thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,
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
