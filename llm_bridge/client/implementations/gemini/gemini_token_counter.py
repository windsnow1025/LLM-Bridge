from google.genai import types


async def count_gemini_tokens(
        response: types.GenerateContentResponse
) -> tuple[int, int]:
    usage_metadata = response.usage_metadata
    if usage_metadata is None:
        return 0, 0
    input_tokens = usage_metadata.prompt_token_count
    if input_tokens is None: # For Vertex AI
        input_tokens = 0
    output_tokens = usage_metadata.candidates_token_count
    if output_tokens is None:
        total_token_count = usage_metadata.total_token_count
        if total_token_count is None: # For Vertex AI
            total_token_count = 0
        output_tokens = total_token_count - input_tokens
    return input_tokens, output_tokens

