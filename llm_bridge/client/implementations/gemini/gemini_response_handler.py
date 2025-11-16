import base64
import mimetypes
from typing import Optional

from google.genai import types
from google.genai.types import Part

from llm_bridge.client.implementations.gemini.gemini_token_counter import count_gemini_tokens
from llm_bridge.client.implementations.printing_status import PrintingStatus
from llm_bridge.type.chat_response import Citation, ChatResponse, File


class GeminiResponseHandler:
    def __init__(self):
        self.printing_status: Optional[PrintingStatus] = None
        self.prev_output_tokens: int = 0
        self.prev_printing_status: Optional[PrintingStatus] = None

    async def process_gemini_response(
            self,
            response: types.GenerateContentResponse,
    ) -> ChatResponse:
        text: str = ""
        thought: str = ""
        code: str = ""
        code_output: str = ""
        files: list[File] = []
        display: Optional[str] = None
        citations: list[Citation] = extract_citations(response)
        input_tokens, stage_output_tokens = await count_gemini_tokens(response)

        parts: list[Part] = []
        if candidates := response.candidates:
            if content := candidates[0].content:
                if content.parts:
                    parts = content.parts

        printing_status: PrintingStatus | None = None
        for part in parts:
            if part.text is not None:
                # Thought
                if part.thought:
                    printing_status = PrintingStatus.Thought
                    thought += part.text
                # Text
                elif not part.thought:
                    printing_status = PrintingStatus.Response
                    text += part.text
            # Code
            if part.executable_code is not None:
                code += part.executable_code.code
            # Code Output
            if part.code_execution_result is not None:
                code_output += part.code_execution_result.output
            # File
            if part.inline_data is not None:
                mime_type = part.inline_data.mime_type
                extension = mimetypes.guess_extension(mime_type) or ""
                file = File(
                    name=f"generated_file{extension}",
                    data=base64.b64encode(part.inline_data.data).decode('utf-8'),
                    type=mime_type,
                )
                files.append(file)

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

        if printing_status == self.prev_printing_status and printing_status == PrintingStatus.Response:
            output_tokens = stage_output_tokens - self.prev_output_tokens
        else:
            output_tokens = stage_output_tokens

        self.prev_output_tokens = stage_output_tokens
        self.prev_printing_status = printing_status

        return ChatResponse(
            text=text,
            thought=thought,
            code=code,
            code_output=code_output,
            files=files,
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
