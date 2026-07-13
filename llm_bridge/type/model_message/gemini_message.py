from enum import StrEnum

from google.genai import types


class GeminiRole(StrEnum):
    User = "user"
    Model = "model"


type GeminiContent = types.Part

type GeminiMessage = types.Content
