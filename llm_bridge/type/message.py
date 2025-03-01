from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    User = "user"
    Assistant = "assistant"
    System = "system"


@dataclass
class Message:
    role: Role
    text: str
    files: list[str]
