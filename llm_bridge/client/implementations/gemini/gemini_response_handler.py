import base64

from google.genai import types

from llm_bridge.client.implementations.gemini.gemini_token_counter import count_gemini_tokens
from llm_bridge.client.implementations.printing_status import PrintingStatus
from llm_bridge.type.chat_response import Citation, ChatResponse


class GeminiResponseHandler:
    def __init__(self):
        self.printing_status = None
        self.prev_output_tokens = 0
        self.prev_printing_status = None

    async def process_gemini_response(
            self,
            response: types.GenerateContentResponse,
    ) -> ChatResponse:
        text = ""
        display = None
        image_base64 = None
        citations = extract_citations(response)
        input_tokens, stage_output_tokens = await count_gemini_tokens(response)

        if candidates := response.candidates:
            if candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # Thought Output
                    if part.text:
                        if part.thought and not self.printing_status:
                            text += "# Model Thought:\n\n"
                            self.printing_status = PrintingStatus.Thought
                        elif not part.thought and self.printing_status == PrintingStatus.Thought:
                            text += f"\n\n# Model Response:\n\n"
                            self.printing_status = PrintingStatus.Response
                        text += part.text
                    # Image Output
                    elif part.inline_data:
                        image_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')

        # Grounding Sources
        if candidates := response.candidates:
            if grounding_metadata := candidates[0].grounding_metadata:
                if search_entry_point := grounding_metadata.search_entry_point:
                    display = search_entry_point.rendered_content
                if grounding_metadata.grounding_chunks:
                    text += "\n\n# Grounding Sources:\n"
                    for i, chunk in enumerate(grounding_metadata.grounding_chunks, start=1):
                        if chunk.web:
                            text += f"{i}. [{chunk.web.title}]({chunk.web.uri})\n"

        if self.printing_status == self.prev_printing_status:
            output_tokens = stage_output_tokens - self.prev_output_tokens
        else:
            output_tokens = stage_output_tokens

        self.prev_output_tokens = stage_output_tokens
        self.prev_printing_status = self.printing_status

        return ChatResponse(
            text=text,
            image=image_base64,
            display=display,
            citations=citations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def extract_citations(response: types.GenerateContentResponse) -> list[Citation]:
    citations = []
    if candidates := response.candidates:
        if grounding_metadata := candidates[0].grounding_metadata:
            if grounding_supports := grounding_metadata.grounding_supports:
                for grounding_support in grounding_supports:
                    citation_indices = [index + 1 for index in grounding_support.grounding_chunk_indices]
                    citation_text = grounding_support.segment.text
                    citations.append(Citation(text=citation_text, indices=citation_indices))
    return citations