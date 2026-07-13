from openai.types.responses import EasyInputMessageParam, ResponseOutputMessageParam, ResponseInputContentParam, \
    ResponseOutputTextParam

from llm_bridge.type.message import Role

type OpenAIResponsesRole = Role

type OpenAIResponsesContent = ResponseInputContentParam | ResponseOutputTextParam

type OpenAIResponsesMessage = EasyInputMessageParam | ResponseOutputMessageParam
