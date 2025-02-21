from dataclasses import dataclass

from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, \
    ChatCompletionContentPartInputAudioParam

from llm_bridge.type.message import Role


@dataclass
class GptMessage:
    role: Role
    content: list[
        ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam | ChatCompletionContentPartInputAudioParam
    ]
