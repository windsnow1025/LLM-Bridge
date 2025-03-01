# LLM Bridge

GitHub: [https://github.com/windsnow1025/LLM-Bridge](https://github.com/windsnow1025/LLM-Bridge)

PyPI: [https://pypi.org/project/LLM-Bridge/](https://pypi.org/project/LLM-Bridge/)

LLM Bridge is a unified Python interface for interacting with LLMs, including OpenAI, OpenAI-Azure, OpenAI-GitHub, Gemini, Claude, and Grok.

## Features
- **Multi-Model Support**: Seamlessly switch between different LLM providers.  
- **Streaming & Non-Streaming**: Supports both real-time streaming and batch responses.  
- **File Processing**: Automatically extracts text from PDFs, Word, Excel, PPT and code files.  
- **Image & Audio Support**: Converts images and audio files into model-compatible formats.  
- **Easy Integration**: Simple API for quick integration into your applications.

## Workflow

LLM Bridge follows a structured process to handle user messages and generate responses:

1. **Message Preprocessor**: Preprocess Messages
    1. **Message Preprocessor**: Extract Text Files to Message
2. **Chat Client Factory**: Create Chat Client
    1. **Model Message Converter**: Convert Message to Model
        1. **Media Processor**: Get Model Image Content from URL
3. **Chat Client**: Generate Response

See `./usage/workflow.py`

## Test

```bash
pytest ./tests/
```

## Installation

### PyPI

```bash
pip install --upgrade llm_bridge
```

## Quick Start

See `./usage/`