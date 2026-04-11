from openai.types.responses import Response, ResponseOutputMessage, \
    ResponseOutputText, ResponseReasoningItem
from openai.types.responses import ResponseStreamEvent, ResponseReasoningSummaryTextDeltaEvent, ResponseTextDeltaEvent, \
    ResponseCodeInterpreterCallCodeDeltaEvent, ResponseImageGenCallPartialImageEvent, ResponseOutputItemDoneEvent, \
    ResponseCodeInterpreterToolCall, ResponseCompletedEvent
from openai.types.responses.response_code_interpreter_tool_call import Output, OutputLogs, OutputImage
from openai.types.responses.response_output_item import ImageGenerationCall

from llm_bridge.logic.chat_generate.media_processor import get_base64_content_from_url
from llm_bridge.type.chat_response import ChatResponse, File


async def process_code_interpreter_outputs(interpreter_outputs: list[Output]) -> tuple[str, list[File]]:
    code_output: str = ""
    files: list[File] = []

    for interpreter_output in interpreter_outputs:
        if isinstance(interpreter_output, OutputLogs):
            code_output += interpreter_output.logs
        if isinstance(interpreter_output, OutputImage):
            data, _ = await get_base64_content_from_url(interpreter_output.url)
            file = File(
                name="code_interpreter_call_output.png",
                data=data,
                type="image/png",
            )
            files.append(file)

    return code_output, files


async def process_openai_responses_non_stream_response(
        response: Response,
) -> ChatResponse:
    text: str = ""
    thought: str = ""
    code: str = ""
    code_output: str = ""
    files: list[File] = []

    for output in response.output:
        if isinstance(output, ResponseOutputMessage):
            for content in output.content:
                if isinstance(content, ResponseOutputText):
                    text += content.text
        elif isinstance(output, ResponseReasoningItem):
            for summary_delta in output.summary:
                thought += summary_delta.text
        elif isinstance(output, ResponseCodeInterpreterToolCall):
            if interpreter_code := output.code:
                code += interpreter_code
            if interpreter_outputs := output.outputs:
                interpreter_code_output, interpreter_files = await process_code_interpreter_outputs(interpreter_outputs)
                code_output += interpreter_code_output
                files.extend(interpreter_files)
        elif isinstance(output, ImageGenerationCall):
            if output.result is not None:
                file = File(
                    name="image_generation_call_output.png",
                    data=output.result,
                    type="image/png",
                )
                files.append(file)

    usage = response.usage
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0
    return ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def process_openai_responses_stream_response(event: ResponseStreamEvent) -> ChatResponse:
    text: str = ""
    thought: str = ""
    code: str = ""
    code_output: str = ""
    files: list[File] = []
    input_tokens: int = 0
    output_tokens: int = 0

    if isinstance(event, ResponseTextDeltaEvent):
        text = event.delta
    elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
        thought = event.delta
    elif isinstance(event, ResponseCodeInterpreterCallCodeDeltaEvent):
        code = event.delta
    elif isinstance(event, ResponseOutputItemDoneEvent):
        if isinstance(event.item, ResponseCodeInterpreterToolCall):
            if interpreter_outputs := event.item.outputs:
                interpreter_code_output, interpreter_files = await process_code_interpreter_outputs(interpreter_outputs)
                code_output += interpreter_code_output
                files.extend(interpreter_files)
    elif isinstance(event, ResponseImageGenCallPartialImageEvent):
        file = File(
            name="image_generation_call_output.png",
            data=event.partial_image_b64,
            type="image/png",
        )
        files.append(file)
    elif isinstance(event, ResponseCompletedEvent):
        usage = event.response.usage
        if usage:
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

    chat_response = ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return chat_response
