from dataclasses import dataclass


@dataclass
class File:
    name: str
    data: str
    type: str


@dataclass
class ChatResponse:
    text: str | None = None
    audio: str | None = None
    thought: str | None = None
    code: str | None = None
    code_output: str | None = None
    files: list[File] | None = None
    display: str | None = None
    error: str | None = None
    input_tokens: int | None = 0
    output_tokens: int | None = 0
