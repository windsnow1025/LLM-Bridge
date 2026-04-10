from dataclasses import dataclass

from llm_bridge import Content, ContentType, Message, Role


@dataclass
class TestConfig:
    name: str
    temperature: float
    stream: bool
    thought: bool
    web_search: bool
    code_execution: bool
    structured_output_schema: dict | None


TimeoutSeconds = 60
MaxRetries = 5
BackoffBase = 1.0

Messages = [
    Message(
        role=Role.User,
        contents=[Content(type=ContentType.Text, data="Hello")]
    )
]

Configs = [
    TestConfig(
        name="basic",
        temperature=0,
        stream=True,
        thought=False,
        web_search=False,
        code_execution=False,
        structured_output_schema=None,
    ),
    TestConfig(
        name="full",
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
]
