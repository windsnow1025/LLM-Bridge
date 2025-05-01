import logging
import re

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError
from openai.types.responses import WebSearchToolParam, Response

from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse, Citation
from llm_bridge.type.serializer import serialize


def process_openai_responses_non_stream_response(
        response: Response
) -> ChatResponse:

    output_list = response.output

    texts: list[str] = []
    citations: list[Citation] = []

    for output in output_list:
        if output.type == "message":
            for content in output.content:
                if content.type == "output_text":
                    texts.append(content.text)
                # Citation is currently not working well in OpenAI Responses API
                if annotations := content.annotations:
                    for annotation in annotations:
                        text = content.text[annotation.start_index:annotation.end_index]

    content = "".join(texts)
    return ChatResponse(text=content, citations=citations)


class NonStreamOpenAIResponsesClient(OpenAIClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")
            response: Response = await self.client.responses.create(
                model=self.model,
                input=serialize(self.messages),
                temperature=self.temperature,
                stream=False,
                tools=self.tools,
            )

            return process_openai_responses_non_stream_response(response)
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