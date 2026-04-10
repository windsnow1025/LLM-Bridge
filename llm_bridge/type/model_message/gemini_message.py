from enum import StrEnum
from typing import TypeAlias

from google.genai import types


class GeminiRole(StrEnum):
    User = "user"
    Model = "model"


GeminiContent: TypeAlias = types.Part

GeminiMessage: TypeAlias = types.Content
