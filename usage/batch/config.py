from dataclasses import dataclass

from llm_bridge import Message
from usage.messages import (
    AudioFileMessages,
    CodeFileMessages,
    DocxFileMessages,
    ImageFileMessages,
    LatencyMessages,
    PdfFileMessages,
    PptxFileMessages,
    TextFileMessages,
    VideoFileMessages,
    XlsxFileMessages,
)


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


TimeoutSeconds = 180
MaxRetries = 4
BackoffBase = 10

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
        name="code_file",
        messages=CodeFileMessages,
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
    TestConfig(
        name="docx_file",
        messages=DocxFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
    TestConfig(
        name="xlsx_file",
        messages=XlsxFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
    TestConfig(
        name="pptx_file",
        messages=PptxFileMessages,
        temperature=0,
        stream=True,
        thought=True,
        web_search=True,
        code_execution=True,
        structured_output_schema=None,
    ),
]
