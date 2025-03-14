from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    User = "user"
    Assistant = "assistant"
    System = "system"


class ContentType(Enum):
    Text = "text"
    File = "file"


@dataclass
class Content:
    type: ContentType
    data: str # text or file path


@dataclass
class Message:
    role: Role
    contents: list[Content]
