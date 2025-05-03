from typing import TypeAlias

from llm_bridge.type.message import Role
from openai.types.responses import EasyInputMessageParam, ResponseOutputMessageParam

OpenAIResponsesRole: TypeAlias = Role


OpenAIResponsesMessage: TypeAlias = EasyInputMessageParam | ResponseOutputMessageParam

# class EasyInputMessageParam(TypedDict, total=False):
#     content: Required[Union[str, ResponseInputMessageContentListParam]]
#     """
#     Text, image, or audio input to the model, used to generate a response. Can also
#     contain previous assistant responses.
#     """
#
#     role: Required[Literal["user", "assistant", "system", "developer"]]
#     """The role of the message input.
#
#     One of `user`, `assistant`, `system`, or `developer`.
#     """
#
#     type: Literal["message"]
#     """The type of the message input. Always `message`."""

# class ResponseOutputMessageParam(TypedDict, total=False):
#     id: Required[str]
#     """The unique ID of the output message."""
#
#     content: Required[Iterable[Content]]
#     """The content of the output message."""
#
#     role: Required[Literal["assistant"]]
#     """The role of the output message. Always `assistant`."""
#
#     status: Required[Literal["in_progress", "completed", "incomplete"]]
#     """The status of the message input.
#
#     One of `in_progress`, `completed`, or `incomplete`. Populated when input items
#     are returned via API.
#     """
#
#     type: Required[Literal["message"]]
#     """The type of the output message. Always `message`."""
