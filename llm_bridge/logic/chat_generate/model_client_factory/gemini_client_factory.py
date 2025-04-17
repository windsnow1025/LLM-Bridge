from google import genai
from google.genai import types
from google.genai._api_client import HttpOptions

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
):
    client = genai.Client(
        api_key=api_key,
        http_options=HttpOptions(api_version='v1alpha') # Thinking
    )

    system_instruction = None
    tools = []
    thinking_config = None
    response_modalities = ['Text']

    if "image" not in model:
        system_instruction = extract_system_messages(messages) or " "
    if "image" not in model:
        tools.append(
            types.Tool(
                google_search=types.GoogleSearch()
            )
        )
    if "image" not in model:
        thinking_config = types.ThinkingConfig(include_thoughts=True)
    if "image" in model:
        response_modalities = ['Text', 'Image']

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
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
