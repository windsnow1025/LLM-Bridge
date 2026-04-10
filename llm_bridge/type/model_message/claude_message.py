from enum import StrEnum
from typing import TypeAlias

from anthropic.types import MessageParam, TextBlockParam, ImageBlockParam, DocumentBlockParam


class ClaudeRole(StrEnum):
    User = "user"
    Assistant = "assistant"


ClaudeContent: TypeAlias = TextBlockParam | ImageBlockParam | DocumentBlockParam

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
