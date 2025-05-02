# LLM Bridge

LLM Bridge is a unified Python interface for interacting with LLMs, including OpenAI, OpenAI-Azure, OpenAI-GitHub, Gemini, Claude, and Grok.

GitHub: [https://github.com/windsnow1025/LLM-Bridge](https://github.com/windsnow1025/LLM-Bridge)

PyPI: [https://pypi.org/project/LLM-Bridge/](https://pypi.org/project/LLM-Bridge/)

## Features
- **Multi-Model Support**: Seamlessly switch between different LLM providers.  
- **Streaming & Non-Streaming**: Supports both real-time streaming and batch responses.  
- **File Processing**: Automatically extracts text from PDFs, Word, Excel, PPT, and code files.  
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

## Test

### Automatic Tests

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

```python
import asyncio
import logging
import os
from pprint import pprint
from typing import AsyncGenerator

from dotenv import load_dotenv

from llm_bridge import *


async def workflow(
        api_keys: dict[str, str],
        messages: list[Message],
        model: str,
        api_type: str,
        temperature: float,
        stream: bool
) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
    await preprocess_messages(messages, api_type)

    chat_client = await create_chat_client(
        messages=messages,
        model=model,
        api_type=api_type,
        temperature=temperature,
        stream=stream,
        api_keys=api_keys,
    )

    if stream:
        return chat_client.generate_stream_response()
    else:
        return await chat_client.generate_non_stream_response()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(".env")

api_keys = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "AZURE_API_KEY": os.environ.get("AZURE_API_KEY"),
    "AZURE_API_BASE": os.environ.get("AZURE_API_BASE"),
    "GITHUB_API_KEY": os.environ.get("GITHUB_API_KEY"),
    "GEMINI_FREE_API_KEY": os.environ.get("GEMINI_FREE_API_KEY"),
    "GEMINI_PAID_API_KEY": os.environ.get("GEMINI_PAID_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "XAI_API_KEY": os.environ.get("XAI_API_KEY"),
}

messages = [
    Message(
        role=Role.System,
        contents=[
            Content(type=ContentType.Text, data="You are a helpful assistant.")
        ]
    ),
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.Text, data="Hello")
        ]
    ),
]
model = "gemini-2.0-flash-exp-image-generation"
api_type = "Gemini-Free"
temperature = 0
stream = True


async def main():
    response = await workflow(api_keys, messages, model, api_type, temperature, stream)
    if stream:
        async for chunk in response:
            pprint(chunk)
    else:
        pprint(response)


if __name__ == "__main__":
    asyncio.run(main())
```

## API

### 1. `preprocess_messages`
Preprocesses a list of messages by extracting text from attached files.

---

### 2. `create_chat_client`
Creates a chat client for the specified LLM provider.

---

### 3. `ChatClient.generate_non_stream_response`
Generates a non-streaming response from the chat model.

---

### 4. `ChatClient.generate_stream_response`
Generates a streaming response from the chat model.

---

### 5. `serialize`
Serializes objects (dataclasses, enums, lists, and dictionaries) into JSON-compatible formats.

---

### 6. `Message`
Represents a message in the conversation.

---

### 7. `ChatResponse`
Represents a response from the chat model.

---

### 8. `Citation`
Represents a citation in a chat response.
