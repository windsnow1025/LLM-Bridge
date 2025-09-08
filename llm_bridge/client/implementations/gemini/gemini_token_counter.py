from google.genai import types


async def count_gemini_tokens(
        response: types.GenerateContentResponse
) -> tuple[int, int]:
    usage_metadata = response.usage_metadata
    if usage_metadata is None:
        return 0, 0
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    if output_tokens is None:
        output_tokens = usage_metadata.total_token_count - usage_metadata.prompt_token_count
    return input_tokens, output_tokens

