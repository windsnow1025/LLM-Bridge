from dataclasses import dataclass
from typing import Optional


@dataclass
class Citation:
    text: str
    indices: list[int]


# TODO: adapt to different Citation formats
# @dataclass
# class Citation:
#     text: str
#     indices: Optional[list[int]] = None
#     url: Optional[str] = None


@dataclass
class ChatResponse:
    text: Optional[str] = None
    thought: Optional[str] = None
    code: Optional[str] = None
    code_output: Optional[str] = None
    image: Optional[str] = None
    display: Optional[str] = None
    citations: Optional[list[Citation]] = None
    error: Optional[str] = None
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
