import logging
import re
from typing import AsyncGenerator

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError, AsyncStream
from openai.types.responses import ResponseStreamEvent, ResponseReasoningSummaryTextDeltaEvent, ResponseTextDeltaEvent, \
    ResponseCodeInterpreterCallCodeDeltaEvent, ResponseImageGenCallPartialImageEvent, ResponseOutputItemDoneEvent, \
    ResponseCodeInterpreterToolCall

from llm_bridge.client.implementations.openai.openai_token_couter import count_openai_responses_input_tokens, \
    count_openai_output_tokens
from llm_bridge.client.implementations.openai.openai_responses_response_handler import process_code_interpreter_outputs
from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse, File
from llm_bridge.type.serializer import serialize


async def process_delta(event: ResponseStreamEvent) -> ChatResponse:
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
    if event.type == "response.image_generation_call.partial_image":
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


async def generate_chunk(
        stream: AsyncStream[ResponseStreamEvent],
        input_tokens: int,
) -> AsyncGenerator[ChatResponse, None]:
    try:
        async for event in stream:
            chat_response = await process_delta(event)
            output_tokens = count_openai_output_tokens(chat_response)
            yield ChatResponse(
                text=chat_response.text,
                thought=chat_response.thought,
                code=chat_response.code,
                code_output=chat_response.code_output,
                files=chat_response.files,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
    except Exception as e:
        logging.exception(e)
        yield ChatResponse(error=repr(e))


class StreamOpenAIResponsesClient(OpenAIClient):
    async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]:
        try:
            logging.info(f"messages: {self.messages}")

            input_tokens = count_openai_responses_input_tokens(
                messages=self.messages
            )

            stream: AsyncStream[ResponseStreamEvent] = await self.client.responses.create(
                model=self.model,
                reasoning=self.reasoning,
                input=serialize(self.messages),
                temperature=self.temperature,
                stream=True,
                tools=self.tools,
                include=self.include,
                # text_format=self.structured_output_base_model, # Async OpenAPI Responses Client does not support structured output
            )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            text = e.response.text
            raise HTTPException(status_code=status_code, detail=text)
        except openai.BadRequestError as e:
            status_code = e.status_code
            text = e.message
            raise HTTPException(status_code=status_code, detail=text)
        except APIStatusError as e:
            status_code = e.status_code
            text = e.message
            raise HTTPException(status_code=status_code, detail=text)
        except Exception as e:
            logging.exception(e)
            match = re.search(r'\d{3}', str(e))
            if match:
                error_code = int(match.group(0))
            else:
                error_code = 500

            raise HTTPException(status_code=error_code, detail=str(e))

        async for chunk in generate_chunk(stream, input_tokens):
            yield chunk