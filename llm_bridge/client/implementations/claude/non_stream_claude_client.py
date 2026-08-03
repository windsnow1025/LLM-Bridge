import logging

from llm_bridge.client.implementations.claude.claude_response_handler import process_claude_non_stream_response
from llm_bridge.client.implementations.http_error import raise_http_exception
from llm_bridge.client.model_client.claude_client import ClaudeClient
from llm_bridge.type.chat_response import ChatResponse
from llm_bridge.type.serializer import serialize


class NonStreamClaudeClient(ClaudeClient):
    async def generate_non_stream_response(self) -> ChatResponse:
        try:
            logging.info(f"messages: {self.messages}")

            message = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                messages=serialize(self.messages),
                betas=self.betas,
                tools=self.tools,
                cache_control=self.cache_control,
                thinking=self.thinking,
                output_config=self.output_config,
            )

            return await process_claude_non_stream_response(
                message=message,
                client=self.client,
            )
        except Exception as e:
            raise_http_exception(e)
