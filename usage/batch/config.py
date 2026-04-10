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
MaxRetries = 8
BackoffBase = 1.0

LatencyMessages = [
    Message(
        role=Role.User,
        contents=[Content(type=ContentType.Text, data="Hello")]
    )
]

TextFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://example-files.online-convert.com/document/txt/example.txt"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

PdfFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://pdfobject.com/pdf/sample.pdf"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

ImageFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://www.gstatic.com/webp/gallery3/1.png"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

AudioFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://samplelib.com/lib/preview/mp3/sample-3s.mp3"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

VideoFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://examplefiles.org/files/video/mp4-example-video-download-640x480.mp4"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

Configs = [
    TestConfig(
        name="latency",
        messages=LatencyMessages,
        temperature=0,
        stream=True,
        thought=False,
        web_search=False,
        code_execution=False,
        structured_output_schema=None,
    ),
    TestConfig(
        name="text_file",
        messages=TextFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
    TestConfig(
        name="pdf_file",
        messages=PdfFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
    TestConfig(
        name="image_file",
        messages=ImageFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
    TestConfig(
        name="audio_file",
        messages=AudioFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
    TestConfig(
        name="video_file",
        messages=VideoFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
]
