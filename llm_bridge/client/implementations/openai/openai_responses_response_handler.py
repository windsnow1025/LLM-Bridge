from openai.types.responses.response_code_interpreter_tool_call import Output, OutputLogs, OutputImage

from llm_bridge.logic.chat_generate.media_processor import get_base64_content_from_url
from llm_bridge.type.chat_response import File


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
