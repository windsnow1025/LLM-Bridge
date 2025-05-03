from google.genai import types


async def count_gemini_tokens(
        response: types.GenerateContentResponse
) -> tuple[int, int]:
    usage_metadata = response.usage_metadata
    return usage_metadata.prompt_token_count, usage_metadata.candidates_token_count

