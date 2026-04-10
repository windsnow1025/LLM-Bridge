from llm_bridge import Content, ContentType, Message, Role

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

CodeFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://raw.githubusercontent.com/windsnow1025/LLM-Bridge/main/pyproject.toml"),
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
            Content(type=ContentType.File, data="https://samplelib.com/wav/sample-3s.wav"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

VideoFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://samplelib.com/mp4/sample-5s.mp4"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

DocxFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://example-files.online-convert.com/document/docx/example.docx"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

XlsxFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://download.microsoft.com/download/1/4/E/14EDED28-6C58-4055-A65C-23B4DA81C4DE/Financial Sample.xlsx"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]

PptxFileMessages = [
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.File, data="https://wiki.documentfoundation.org/images/4/47/Extlst-test.pptx"),
            Content(type=ContentType.Text, data="What's this?"),
        ]
    )
]
