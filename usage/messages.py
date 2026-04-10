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
