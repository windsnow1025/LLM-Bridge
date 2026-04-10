from typing import TypeAlias

from openai.types.responses import EasyInputMessageParam, ResponseOutputMessageParam, ResponseInputContentParam, \
    ResponseOutputTextParam

from llm_bridge.type.message import Role

OpenAIResponsesRole: TypeAlias = Role

OpenAIResponsesContent: TypeAlias = ResponseInputContentParam | ResponseOutputTextParam

OpenAIResponsesMessage: TypeAlias = EasyInputMessageParam | ResponseOutputMessageParam
