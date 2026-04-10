from dataclasses import dataclass

from llm_bridge import Content, ContentType, Message, Role


@dataclass
class TestConfig:
    name: str
    messages: list[Message]
    temperature: float
    stream: bool
    thought: bool
    web_search: bool
    code_execution: bool
    structured_output_schema: dict | None


TimeoutSeconds = 60
MaxRetries = 5
BackoffBase = 1.0

BasicMessages = [
    Message(
        role=Role.User,
        contents=[Content(type=ContentType.Text, data="Hello")]
    )
]

FullMessages = [
    Message(
        role=Role.System,
        contents=[Content(type=ContentType.Text, data="You are a helpful assistant.")]
    ),
    Message(
        role=Role.User,
        contents=[Content(type=ContentType.Text, data="Hello")]
    ),
    Message(
        role=Role.Assistant,
        contents=[Content(type=ContentType.Text, data="Hello! How can I assist you today?")]
    ),
    Message(
        role=Role.User,
        contents=[Content(type=ContentType.Text, data="Say this is a test.")]
    ),
]

Configs = [
    TestConfig(
        name="basic",
        messages=BasicMessages,
        temperature=0,
        stream=True,
        thought=False,
        web_search=False,
        code_execution=False,
        structured_output_schema=None,
    ),
    TestConfig(
        name="full",
        messages=FullMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
]
