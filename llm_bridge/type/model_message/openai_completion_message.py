from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam

from llm_bridge.type.message import Role

type OpenAICompletionRole = Role

type OpenAICompletionContent = ChatCompletionContentPartParam

# type OpenAICompletionMessage = ChatCompletionMessageParam
type OpenAICompletionMessage = ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam
