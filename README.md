# LLM Bridge

LLM Bridge is a unified Python interface for interacting with LLMs, including OpenAI, OpenAI-Azure, OpenAI-GitHub, Gemini, Claude, and Grok.

GitHub: [https://github.com/windsnow1025/LLM-Bridge](https://github.com/windsnow1025/LLM-Bridge)

PyPI: [https://pypi.org/project/LLM-Bridge/](https://pypi.org/project/LLM-Bridge/)

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

## Test

### Automatic Tests

```bash
pytest ./tests/
```

### Manual Tests

See `./usage/`

## Installation

### PyPI

```bash
pip install --upgrade llm_bridge
```

## Quick Start

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
    await preprocess_messages(messages)

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
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "XAI_API_KEY": os.environ.get("XAI_API_KEY"),
}

messages = [
    Message(
        role=Role.System,
        text="You are a helpful assistant.",
        files=[]
    ),
    Message(
        role=Role.User,
        text="Hello",
        files=[]
    )
]
model = "gemini-2.0-flash-001"
api_type = "Gemini"  # OpenAI / OpenAI-Azure / OpenAI-GitHub / Gemini / Claude / Grok
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

#### **Function Signature**
```python
async def preprocess_messages(messages: list[Message]) -> None
```

#### **Parameters**
- `messages` (`list[Message]`): A list of `Message` objects to be preprocessed.

#### **Returns**
- `None`: Modifies the `messages` list in place.

---

### 2. `create_chat_client`
Creates a chat client for the specified LLM provider.

#### **Function Signature**
```python
async def create_chat_client(
    messages: list[Message],
    model: str,
    api_type: str,
    temperature: float,
    stream: bool,
    api_keys: dict
) -> ChatClient
```

#### **Parameters**
- `messages` (`list[Message]`): A list of messages to be sent to the model.
- `model` (`str`): The model name (e.g., `"gpt-4"`, `"gemini-2.0-flash-001"`).
- `api_type` (`str`): The API provider (`"OpenAI"`, `"OpenAI-Azure"`, `"OpenAI-GitHub"`, `"Gemini"`, `"Claude"`, `"Grok"`).
- `temperature` (`float`): The temperature setting for response randomness.
- `stream` (`bool`): Whether to use streaming responses.
- `api_keys` (`dict`): A dictionary containing API keys for the selected provider.

#### **Returns**
- `ChatClient`: A chat client instance for generating responses.

---

### 3. `ChatClient.generate_non_stream_response`
Generates a non-streaming response from the chat model.

#### **Function Signature**
```python
async def generate_non_stream_response(self) -> ChatResponse
```

#### **Returns**
- `ChatResponse`: A response object containing the generated text.

---

### 4. `ChatClient.generate_stream_response`
Generates a streaming response from the chat model.

#### **Function Signature**
```python
async def generate_stream_response(self) -> AsyncGenerator[ChatResponse, None]
```

#### **Returns**
- `AsyncGenerator[ChatResponse, None]`: A generator yielding `ChatResponse` objects.

---

### 5. `serialize`
Serializes objects (dataclasses, enums, lists, and dictionaries) into JSON-compatible formats.

#### **Function Signature**
```python
def serialize(obj: Any) -> str | dict | list | Any
```

#### **Parameters**
- `obj` (`Any`): The object to serialize.

#### **Returns**
- `str | dict | list | Any`: A serialized representation of the object.

---

### 6. `Message`
Represents a message in the conversation.

#### **Class Definition**
```python
@dataclass
class Message:
    role: Role
    text: str
    files: list[str]
```

#### **Attributes**
- `role` (`Role`): The role of the message sender (`User`, `Assistant`, `System`).
- `text` (`str`): The message content.
- `files` (`list[str]`): A list of file URLs attached to the message.

---

### 7. `ChatResponse`
Represents a response from the chat model.

#### **Class Definition**
```python
@dataclass
class ChatResponse:
    text: Optional[str] = None
    display: Optional[str] = None
    citations: Optional[list[Citation]] = None
    error: Optional[str] = None
```

#### **Attributes**
- `text` (`Optional[str]`): The generated response text.
- `display` (`Optional[str]`): A formatted version of the response.
- `citations` (`Optional[list[Citation]]`): A list of citations (if applicable).
- `error` (`Optional[str]`): An error message (if any).

---

### 8. `Citation`
Represents a citation in a chat response.

#### **Class Definition**
```python
@dataclass
class Citation:
    text: str
    indices: list[int]
```

#### **Attributes**
- `text` (`str`): The citation text.
- `indices` (`list[int]`): The indices where the citation applies.
