from google.genai import types


async def count_gemini_tokens(
        response: types.GenerateContentResponse
) -> tuple[int, int]:
    usage = response.usage_metadata
    if usage is None:
        return 0, 0
    input_tokens = usage.prompt_token_count or 0
    total_tokens = usage.total_token_count or 0
    output_tokens = total_tokens - input_tokens
    return input_tokens, output_tokens
