import logging
import re

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError
from openai.types.responses import Response

from llm_bridge.client.implementations.openai.openai_responses_response_handler import \
    process_openai_responses_non_stream_response
from llm_bridge.client.implementations.openai.openai_token_couter import count_openai_responses_input_tokens
from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


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
                include=self.include,
                # text_format=self.structured_output_base_model, # Async OpenAPI Responses Client does not support structured output
            )

            return await process_openai_responses_non_stream_response(
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