from dataclasses import dataclass
from enum import Enum

from anthropic.types import MessageParam
from typing_extensions import TypeAlias

from llm_bridge.type.message import Role


class ClaudeRole(Enum):
    User = "user"
    Assistant = "assistant"


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
