import mimetypes

from google.genai.interactions import Interaction, ModelOutputStep, TextContent, ThoughtStep, CodeExecutionCallStep, \
    CodeExecutionResultStep, ImageContent, GoogleSearchResultStep, URLCitation, InteractionSSEEvent, StepDelta, \
    TextDelta, ThoughtSummaryDelta, CodeExecutionCallDelta, CodeExecutionResultDelta, ImageDelta, \
    GoogleSearchResultDelta, TextAnnotationDelta, InteractionCompletedEvent

from llm_bridge.type.chat_response import ChatResponse, File


async def process_gemini_interactions_non_stream_response(
        interaction: Interaction,
) -> ChatResponse:
    text: str = ""
    thought: str = ""
    code: str = ""
    code_output: str = ""
    files: list[File] = []
    display: str = ""
    citations: dict[str, str] = {}

    for step in interaction.steps:
        if isinstance(step, ModelOutputStep):
            for content in step.content:
                if isinstance(content, TextContent):
                    text += content.text
                    if content.annotations:
                        for annotation in content.annotations:
                            if isinstance(annotation, URLCitation):
                                citations[annotation.url] = annotation.title
                elif isinstance(content, ImageContent):
                    if content.mime_type:
                        extension = mimetypes.guess_extension(content.mime_type) or ""
                        file = File(
                            name=f"generated_file{extension}",
                            data=content.data,
                            type=content.mime_type,
                        )
                        files.append(file)
        elif isinstance(step, ThoughtStep):
            if step.summary:
                for summary_content in step.summary:
                    if isinstance(summary_content, TextContent):
                        thought += summary_content.text
        elif isinstance(step, CodeExecutionCallStep):
            if step.arguments.code:
                code += step.arguments.code
        elif isinstance(step, CodeExecutionResultStep):
            code_output += step.result
        elif isinstance(step, GoogleSearchResultStep):
            for search_result in step.result:
                if search_result.search_suggestions:
                    display += search_result.search_suggestions

    if citations:
        text += "\n\n# Grounding Sources:\n"
        for i, (url, title) in enumerate(citations.items(), start=1):
            text += f"{i}. [{title}]({url})\n"

    usage = interaction.usage
    input_tokens = (usage.total_input_tokens or 0) + (usage.total_tool_use_tokens or 0)
    output_tokens = (usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0)

    return ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
        display=display,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class GeminiInteractionsStreamResponseHandler:
    def __init__(self):
        self.citations: dict[str, str] = {} # title: url

    async def process_gemini_interactions_stream_response(self, event: InteractionSSEEvent) -> ChatResponse:
        text: str = ""
        thought: str = ""
        code: str = ""
        code_output: str = ""
        files: list[File] = []
        display: str = ""
        input_tokens: int = 0
        output_tokens: int = 0

        if isinstance(event, StepDelta):
            delta = event.delta
            if isinstance(delta, TextDelta):
                text = delta.text
            elif isinstance(delta, ThoughtSummaryDelta):
                if isinstance(delta.content, TextContent):
                    thought = delta.content.text
            elif isinstance(delta, CodeExecutionCallDelta):
                if delta.arguments.code:
                    code = delta.arguments.code
            elif isinstance(delta, CodeExecutionResultDelta):
                code_output = delta.result
            elif isinstance(delta, ImageDelta):
                if delta.mime_type:
                    extension = mimetypes.guess_extension(delta.mime_type) or ""
                    file = File(
                        name=f"generated_file{extension}",
                        data=delta.data,
                        type=delta.mime_type,
                    )
                    files.append(file)
            elif isinstance(delta, GoogleSearchResultDelta):
                for search_result in delta.result:
                    if search_result.search_suggestions:
                        display += search_result.search_suggestions
            elif isinstance(delta, TextAnnotationDelta):
                if delta.annotations:
                    for annotation in delta.annotations:
                        if isinstance(annotation, URLCitation):
                            self.citations[annotation.url] = annotation.title
        elif isinstance(event, InteractionCompletedEvent):
            if self.citations:
                text += "\n\n# Grounding Sources:\n"
                for i, (url, title) in enumerate(self.citations.items(), start=1):
                    text += f"{i}. [{title}]({url})\n"
            usage = event.interaction.usage
            input_tokens = (usage.total_input_tokens or 0) + (usage.total_tool_use_tokens or 0)
            output_tokens = (usage.total_output_tokens or 0) + (usage.total_thought_tokens or 0)

        return ChatResponse(
            text=text,
            thought=thought,
            code=code,
            code_output=code_output,
            files=files,
            display=display,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
