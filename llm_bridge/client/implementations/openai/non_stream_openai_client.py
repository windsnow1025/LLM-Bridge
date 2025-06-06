import logging
import re

import httpx
import openai
from fastapi import HTTPException
from openai import APIStatusError

from llm_bridge.client.implementations.openai.openai_token_couter import count_openai_input_tokens, \
    count_openai_output_tokens
from llm_bridge.client.model_client.openai_client import OpenAIClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


class NonStreamOpenAIClient(OpenAIClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            input_tokens = count_openai_input_tokens(
                messages=self.messages
            )

            completion = await self.client.chat.completions.create(
                messages=serialize(self.messages),
                model=self.model,
                temperature=self.temperature,
                stream=False
            )

            content = completion.choices[0].message.content
            chat_response = ChatResponse(text=content)
            output_tokens = count_openai_output_tokens(chat_response)
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
