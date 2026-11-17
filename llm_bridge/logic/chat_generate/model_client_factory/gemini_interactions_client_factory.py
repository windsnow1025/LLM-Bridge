from typing import Any

from google import genai
from google.genai.interactions import ToolParam, GoogleSearchParam, URLContextParam, CodeExecutionParam, \
    GenerationConfigParam, SafetySettingParam, ServiceTier, TextResponseFormatParam

from llm_bridge.client.implementations.gemini_interactions.non_stream_gemini_interactions_client import \
    NonStreamGeminiInteractionsClient
from llm_bridge.client.implementations.gemini_interactions.stream_gemini_interactions_client import \
    StreamGeminiInteractionsClient
from llm_bridge.logic.chat_generate.chat_message_converter import convert_messages_to_gemini_interactions
from llm_bridge.logic.message_process.message_processor import extract_system_message, contains_audio_or_video
from llm_bridge.type.message import Message


async def create_gemini_interactions_client(
        api_key: str,
        free_tier: bool,
        messages: list[Message],
        model: str,
        temperature: float,
        stream: bool,
        thought: bool,
        web_search: bool,
        code_execution: bool,
        structured_output_schema: dict[str, Any] | None,
) -> StreamGeminiInteractionsClient | NonStreamGeminiInteractionsClient:
    client = genai.Client(
        vertexai=False,
        api_key=api_key,
    )

    system_instruction = extract_system_message(messages)
    tools: list[ToolParam] = []
    generation_config = GenerationConfigParam()
    safety_settings: list[SafetySettingParam] = [
        SafetySettingParam(type="hate_speech", threshold="off"),
        SafetySettingParam(type="dangerous_content", threshold="off"),
        SafetySettingParam(type="harassment", threshold="off"),
        SafetySettingParam(type="sexually_explicit", threshold="off"),
        SafetySettingParam(type="jailbreak", threshold="off"),
    ]
    service_tier: ServiceTier = "standard"
    response_format = TextResponseFormatParam(type="text")

    if web_search and not free_tier:
        tools.append(GoogleSearchParam(type="google_search"))
    if "image" not in model:
        tools.append(URLContextParam(type="url_context"))
        if code_execution and not await contains_audio_or_video(messages):
            tools.append(CodeExecutionParam(type="code_execution"))
    if "image" in model:
        generation_config["temperature"] = temperature
    if thought:
        generation_config["thinking_level"] = "high"
        generation_config["thinking_summaries"] = "auto"
    if not free_tier:
        service_tier = "priority"
    if structured_output_schema:
        response_format["mime_type"] = "application/json"
        response_format["schema_"] = structured_output_schema

    gemini_messages = await convert_messages_to_gemini_interactions(messages)

    if "image" not in model:
        for gemini_message in gemini_messages:
            for content in gemini_message["content"]:
                if content["type"] in ("image", "video"):
                    content["resolution"] = "high"

    if stream:
        return StreamGeminiInteractionsClient(
            model=model,
            system_instruction=system_instruction,
            messages=gemini_messages,
            client=client,
            tools=tools,
            generation_config=generation_config,
            safety_settings=safety_settings,
            service_tier=service_tier,
            response_format=response_format,
        )
    else:
        return NonStreamGeminiInteractionsClient(
            model=model,
            system_instruction=system_instruction,
            messages=gemini_messages,
            client=client,
            tools=tools,
            generation_config=generation_config,
            safety_settings=safety_settings,
            service_tier=service_tier,
            response_format=response_format,
        )
