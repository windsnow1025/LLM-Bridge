# https://ai.google.dev/gemini-api/docs/file-input-methods
GeminiDocumentMimeTypes = {
    "text/html",
    "text/css",
    "text/plain",
    "text/xml",
    "text/csv",
    "text/rtf",
    "text/javascript",

    "application/json",
    "application/pdf",
}

GeminiImageMimeTypes = {
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
}

# https://ai.google.dev/gemini-api/docs/audio#supported-formats
GeminiAudioMimeTypes = {
    "audio/wav",
    "audio/mp3",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
}

# https://ai.google.dev/gemini-api/docs/video-understanding#supported-formats
GeminiVideoMimeTypes = {
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
}

GeminiSupportedMimeTypes = GeminiDocumentMimeTypes | GeminiImageMimeTypes | GeminiAudioMimeTypes | GeminiVideoMimeTypes
