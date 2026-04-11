import logging
import re

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError
from openai.types.chat import ChatCompletion

from llm_bridge.client.model_client.openai_completion_client import OpenAICompletionClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


class NonStreamOpenAICompletionClient(OpenAICompletionClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            completion: ChatCompletion = await self.client.chat.completions.create(
                messages=serialize(self.messages),
                model=self.model,
                temperature=self.temperature,
                stream=False,
                reasoning_effort=self.reasoning_effort,
                response_format=self.response_format,
            )

            content = completion.choices[0].message.content
            usage = completion.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            return ChatResponse(
                text=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
