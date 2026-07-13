from enum import StrEnum

from anthropic.types import MessageParam, TextBlockParam, ImageBlockParam, DocumentBlockParam


class ClaudeRole(StrEnum):
    User = "user"
    Assistant = "assistant"


type ClaudeContent = TextBlockParam | ImageBlockParam | DocumentBlockParam

type ClaudeMessage = MessageParam

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
