from enum import Enum

from anthropic.types import MessageParam, TextBlockParam, ImageBlockParam, DocumentBlockParam
from typing_extensions import TypeAlias


class ClaudeRole(Enum):
    User = "user"
    Assistant = "assistant"


ClaudeContent = TextBlockParam | ImageBlockParam | DocumentBlockParam
ClaudeMessage: TypeAlias = MessageParam

# class MessageParam(TypedDict, total=False):
#     content: Required[
#         Union[
#             str,
#             Iterable[
#                 Union[
#                     TextBlockParam,
#                     ImageBlockParam,
#                     ToolUseBlockParam,
#                     ToolResultBlockParam,
#                     DocumentBlockParam,
#                     ThinkingBlockParam,
#                     RedactedThinkingBlockParam,
#                     ContentBlock,
#                 ]
#             ],
#         ]
#     ]
#
#     role: Required[Literal["user", "assistant"]]
