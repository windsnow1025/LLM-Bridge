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
            Content(type=ContentType.File, data="https://samplelib.com/mp3/sample-3s.mp3"),
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
