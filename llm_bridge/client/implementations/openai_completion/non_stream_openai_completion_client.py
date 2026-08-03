import logging

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionAudio

from llm_bridge.client.implementations.http_error import raise_http_exception
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
                modalities=self.modalities,
                audio=self.audio,
                response_format=self.response_format,
            )

            message: ChatCompletionMessage = completion.choices[0].message
            content: str | None = message.content
            audio_data: str | None = None
            if message.audio is not None:
                audio: ChatCompletionAudio = message.audio
                content = audio.transcript
                audio_data = audio.data
            usage = completion.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            return ChatResponse(
                text=content,
                audio=audio_data,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:
            raise_http_exception(e)
