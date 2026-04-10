from typing import TypeAlias

from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam, \
    ChatCompletionContentPartInputAudioParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)

from llm_bridge.type.message import Role

OpenAICompletionRole: TypeAlias = Role

# OpenAICompletionContent: TypeAlias = ChatCompletionContentPartParam
OpenAICompletionContent: TypeAlias = ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam | ChatCompletionContentPartInputAudioParam

# OpenAICompletionMessage: TypeAlias = ChatCompletionMessageParam
OpenAICompletionMessage: TypeAlias = ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam
