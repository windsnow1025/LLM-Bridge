from dataclasses import dataclass
from typing import Optional


@dataclass
class File:
    name: str
    data: str # Base64
    type: str


@dataclass
class ChatResponse:
    text: Optional[str] = None
    thought: Optional[str] = None
    code: Optional[str] = None
    code_output: Optional[str] = None
    files: Optional[list[File]] = None
    display: Optional[str] = None
    error: Optional[str] = None
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
