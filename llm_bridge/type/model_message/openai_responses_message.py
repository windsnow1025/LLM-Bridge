from typing import TypeAlias

from llm_bridge.type.message import Role
from openai.types.responses import EasyInputMessageParam


OpenAIResponsesRole: TypeAlias = Role


OpenAIResponsesMessage: TypeAlias = EasyInputMessageParam

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