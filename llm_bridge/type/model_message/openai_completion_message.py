from typing import TypeAlias

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam

from llm_bridge.type.message import Role

OpenAICompletionRole: TypeAlias = Role

OpenAICompletionContent: TypeAlias = ChatCompletionContentPartParam

# OpenAICompletionMessage: TypeAlias = ChatCompletionMessageParam
OpenAICompletionMessage: TypeAlias = ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam
