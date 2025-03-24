import base64
from io import BytesIO
from enum import Enum
from pprint import pprint

from google.genai import types
from llm_bridge.type.chat_response import Citation, ChatResponse


class PrintingStatus(Enum):
    Start = "start"
    Thought = "thought"
    Response = "response"


class GeminiResponseHandler:
    def __init__(self):
        self.printing_status = PrintingStatus.Start

    def process_gemini_response(
            self, response: types.GenerateContentResponse
    ) -> ChatResponse:
        text = ""
        display = None
        image_base64 = None
        citations = extract_citations(response)

        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    if part.thought and self.printing_status == PrintingStatus.Start:
                        text += "# Model Thought:\n\n"
                        self.printing_status = PrintingStatus.Thought
                    elif not part.thought and self.printing_status == PrintingStatus.Thought:
                        text += f"\n\n# Model Response:\n\n"
                        self.printing_status = PrintingStatus.Response
                    text += part.text
                elif part.inline_data:
                    image_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')

        if grounding_metadata := response.candidates[0].grounding_metadata:
            if search_entry_point := grounding_metadata.search_entry_point:
                display = search_entry_point.rendered_content
            if grounding_metadata.grounding_chunks:
                text += "\n\n# Grounding Sources:\n"
                for i, chunk in enumerate(grounding_metadata.grounding_chunks, start=1):
                    if chunk.web:
                        text += f"{i}. [{chunk.web.title}]({chunk.web.uri})\n"

        return ChatResponse(text=text, image=image_base64, display=display, citations=citations)


def extract_citations(response: types.GenerateContentResponse) -> list[Citation]:
    citations = []
    if grounding_metadata := response.candidates[0].grounding_metadata:
        if grounding_supports := grounding_metadata.grounding_supports:
            for grounding_support in grounding_supports:
                citation_indices = [index + 1 for index in grounding_support.grounding_chunk_indices]
                citation_text = grounding_support.segment.text
                citations.append(Citation(text=citation_text, indices=citation_indices))
    return citations if citations else None