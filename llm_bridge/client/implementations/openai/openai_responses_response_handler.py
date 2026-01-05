from openai.types.responses import Response, ResponseOutputItem, ResponseOutputMessage, \
    ResponseOutputText, ResponseReasoningItem
from openai.types.responses import ResponseStreamEvent, ResponseReasoningSummaryTextDeltaEvent, ResponseTextDeltaEvent, \
    ResponseCodeInterpreterCallCodeDeltaEvent, ResponseImageGenCallPartialImageEvent, ResponseOutputItemDoneEvent, \
    ResponseCodeInterpreterToolCall
from openai.types.responses.response_code_interpreter_tool_call import Output, OutputLogs, OutputImage
from openai.types.responses.response_output_item import ImageGenerationCall

from llm_bridge.client.implementations.openai.openai_token_couter import count_openai_output_tokens
from llm_bridge.logic.chat_generate.media_processor import get_base64_content_from_url
from llm_bridge.type.chat_response import ChatResponse, File


async def process_code_interpreter_outputs(interpreter_outputs: list[Output]) -> tuple[str, list[File]]:
    code_output: str = ""
    files: list[File] = []

    for interpreter_output in interpreter_outputs:
        if interpreter_output.type == "logs":
            output_logs: OutputLogs = interpreter_output
            code_output += output_logs.logs
        if interpreter_output.type == "image":
            output_image: OutputImage = interpreter_output
            data, _ = await get_base64_content_from_url(output_image.url)
            file = File(
                name="code_interpreter_call_output.png",
                data=data,
                type="image/png",
            )
            files.append(file)

    return code_output, files


async def process_openai_responses_non_stream_response(
        response: Response,
        input_tokens: int,
) -> ChatResponse:

    output_list: list[ResponseOutputItem] = response.output

    text: str = ""
    thought: str = ""
    code: str = ""
    code_output: str = ""
    files: list[File] = []

    for output in output_list:
        if output.type == "message":
            output_message: ResponseOutputMessage = output
            for content in output_message.content:
                if content.type == "output_text":
                    output_text: ResponseOutputText = content
                    text += output_text.text
        elif output.type == "reasoning":
            reasoning_item: ResponseReasoningItem = output
            for summary_delta in reasoning_item.summary:
                thought += summary_delta.text
        elif output.type == "code_interpreter_call":
            code_interpreter_tool_call: ResponseCodeInterpreterToolCall = output
            if interpreter_code := code_interpreter_tool_call.code:
                code += interpreter_code
            if interpreter_outputs := code_interpreter_tool_call.outputs:
                interpreter_code_output, interpreter_files = await process_code_interpreter_outputs(interpreter_outputs)
                code_output += interpreter_code_output
                files.extend(interpreter_files)
        elif output.type == "image_generation_call":
            image_generation_call: ImageGenerationCall = output
            file = File(
                name="image_generation_call_output.png",
                data=image_generation_call.result,
                type="image/png",
            )
            files.append(file)

    chat_response = ChatResponse(text=text, files=files)
    output_tokens = count_openai_output_tokens(chat_response)
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

    if event.type == "response.output_text.delta":
        text_delta_event: ResponseTextDeltaEvent = event
        text = text_delta_event.delta
    elif event.type == "response.reasoning_summary_text.delta":
        reasoning_summary_text_delta_event: ResponseReasoningSummaryTextDeltaEvent = event
        thought = reasoning_summary_text_delta_event.delta
    elif event.type == "response.code_interpreter_call_code.delta":
        code_interpreter_call_code_delta_event: ResponseCodeInterpreterCallCodeDeltaEvent = event
        code = code_interpreter_call_code_delta_event.delta
    elif event.type == "response.output_item.done":
        output_item_done_event: ResponseOutputItemDoneEvent = event
        if output_item_done_event.item.type == "code_interpreter_call":
            code_interpreter_tool_call: ResponseCodeInterpreterToolCall = output_item_done_event.item
            if interpreter_outputs := code_interpreter_tool_call.outputs:
                interpreter_code_output, interpreter_files = await process_code_interpreter_outputs(interpreter_outputs)
                code_output += interpreter_code_output
                files.extend(interpreter_files)
    elif event.type == "response.image_generation_call.partial_image":
        image_gen_call_partial_image_event: ResponseImageGenCallPartialImageEvent = event
        file = File(
            name="generated_image.png",
            data=image_gen_call_partial_image_event.partial_image_b64,
            type="image/png",
        )
        files.append(file)

    chat_response = ChatResponse(
        text=text,
        thought=thought,
        code=code,
        code_output=code_output,
        files=files,
    )
    return chat_response
