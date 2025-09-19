import logging
import re
from pprint import pprint
from typing import Optional

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError
from openai.types.responses import WebSearchToolParam, Response

from llm_bridge.client.implementations.openai.openai_token_couter import count_openai_responses_input_tokens, \
    count_openai_output_tokens
from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse, Citation, File
from llm_bridge.type.serializer import serialize


def process_openai_responses_non_stream_response(
        response: Response,
        input_tokens: int,
) -> ChatResponse:

    output_list = response.output

    text: str = ""
    files: list[File] = []
    citations: list[Citation] = []

    for output in output_list:
        if output.type == "message":
            for content in output.content:
                if content.type == "output_text":
                    text += content.text
                # Citation is unavailable in OpenAI Responses API
                # if annotations := content.annotations:
                #     for annotation in annotations:
                #         citations.append(
                #             Citation(
                #                 text=content.text[annotation.start_index:annotation.end_index],
                #                 url=annotation.url
                #             )
                #         )
        # Image Generation untestable due to organization verification requirement
        # if output.type == "image_generation_call":
        #     file = File(
        #         name="generated_image.png",
        #         data=output.result,
        #         type="image/png",
        #     )
        #     files.append(file)

    chat_response = ChatResponse(text=text, files=files)
    output_tokens = count_openai_output_tokens(chat_response)
    return ChatResponse(
        text=text,
        files=files,
        citations=citations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class NonStreamOpenAIResponsesClient(OpenAIClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            input_tokens = count_openai_responses_input_tokens(
                messages=self.messages
            )

            response: Response = await self.client.responses.create(
                model=self.model,
                reasoning=self.reasoning,
                input=serialize(self.messages),
                temperature=self.temperature,
                stream=False,
                tools=self.tools,
            )

            return process_openai_responses_non_stream_response(
                response=response,
                input_tokens=input_tokens,
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