from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    User = "user"
    Assistant = "assistant"
    System = "system"


class ContentType(StrEnum):
    Text = "text"
    File = "file"


@dataclass
class Content:
    type: ContentType
    data: str # Markdown Text or File Url


@dataclass
class Message:
    role: Role
    contents: list[Content]
