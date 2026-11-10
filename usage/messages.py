from llm_bridge import Content, ContentType, Message, Role
from usage.example_file_urls import (
    TextFileUrl,
    CodeFileUrl,
    PdfFileUrl,
    ImageFileUrl,
    AudioFileUrl,
    VideoFileUrl,
    DocxFileUrl,
    XlsxFileUrl,
    PptxFileUrl,
)

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
            Content(type=ContentType.File, data=TextFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

CodeFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=CodeFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

PdfFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=PdfFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

ImageFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=ImageFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

AudioFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=AudioFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

VideoFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=VideoFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

DocxFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=DocxFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

XlsxFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=XlsxFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

PptxFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data=PptxFileUrl),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]
