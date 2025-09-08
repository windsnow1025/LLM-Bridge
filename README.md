# LLM Bridge

LLM Bridge is a unified Python interface for interacting with LLMs, including OpenAI, OpenAI-Azure, OpenAI-GitHub, Gemini, Claude, and Grok.

GitHub: [https://github.com/windsnow1025/LLM-Bridge](https://github.com/windsnow1025/LLM-Bridge)

PyPI: [https://pypi.org/project/LLM-Bridge/](https://pypi.org/project/LLM-Bridge/)

## Workflow and Features

1. **Message Preprocessor**: extracts text content from documents (Word, Excel, PPT, Code files, PDFs) which are not natively supported by the target model.
2. **Chat Client Factory**: create a client for the specific LLM API with model parameters
    1. **Model Message Converter**: convert general messages to model messages
        1. **Media Processor**: converts media (Image, Audio, Video, PDF) which are natively supported by the target model into compatible formats.
3. **Chat Client**: generate stream or non-stream responses
    1. **Model Thoughts**: captures and formats the model's thinking process
    2. **Search Citations**: extracts and formats citations from search results
    3. **Token Counter**: tracks and reports input and output token usage

### Model Features

The features listed represent the maximum capabilities of each API type supported by LLM Bridge.

| API Type | Input Format                   | Capabilities                                               | Output Format |
|----------|--------------------------------|------------------------------------------------------------|---------------|
| OpenAI   | Text, Image                    | Thinking, Web Search, Code Execution                       | Text          |
| Gemini   | Text, Image, Video, Audio, PDF | Thinking + Thought, Web Search + Citations, Code Execution | Text, Image   |
| Claude   | Text, Image, PDF               | Thinking, Web Search                                       | Text          |
| Grok     | Text, Image                    |                                                            | Text          |

#### Planned Features

- OpenAI: Web Search: Citations, Image Output
- Gemini: Code Execution: Code, Code Output
- Claude: Code Execution, File Output

## Installation

```bash
pip install --upgrade llm_bridge
```

## Test

```bash
pytest
```

## Quick Start

### Setup

1. Copy `./usage/.env.example` and rename it to `./usage/.env`, then fill in the environment variables.
2. Install requirements: `pip install -r requirements.txt`
3. In PyCharm, add a new Python configuration:
   - script: `./usage/main.py`
   - Paths to ".env" files: `./usage/.env`

## Workflow

```python
from typing import AsyncGenerator

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
```

### Main

```python
import asyncio
import logging
import os
from pprint import pprint

from dotenv import load_dotenv

from llm_bridge import *
from usage.workflow import workflow

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
    Message(
        role=Role.Assistant,
        contents=[
            Content(type=ContentType.Text, data="Hello! How can I assist you today?")
        ]
    ),
    Message(
        role=Role.User,
        contents=[
            Content(type=ContentType.Text, data="Explain the concept of Occam's Razor and provide a simple, everyday example."),
            # Content(type=ContentType.Text, data="What's the weather in NYC today?"),
            # Content(type=ContentType.Text, data="Please generate an image of a cat."),
        ]
    ),
    # Message(
    #     role=Role.User,
    #     contents=[
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746208707489-image.png"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746209841847-A%20Tutorial%20on%20Spectral%20Clustering.pdf"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746212253473-file_example_MP3_700KB.mp3"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746212980820-file_example_MP4_480_1_5MG.mp4"),
    #         Content(type=ContentType.Text, data="What's this?"),
    #     ]
    # ),
]
# See /llm_bridge/resources/model_prices.json for available models
# model = "gpt-4.1"
# model = "gemini-2.5-flash-preview-native-audio-dialog"
# model = "gemini-2.5-pro-exp-03-25"
model = "gemini-2.5-pro-preview-05-06"
# model = "claude-sonnet-4-0"
# api_type = "OpenAI"
# api_type = "Gemini-Free"
api_type = "Gemini-Paid"
# api_type = "Claude"
temperature = 0
stream = True


async def main():
    model_prices = get_model_prices()
    pprint(model_prices)

    input_tokens = 0
    output_tokens = 0
    response = await workflow(api_keys, messages, model, api_type, temperature, stream)
    text = ""
    if stream:
        async for chunk in response:
            pprint(chunk)
            if chunk.text:
                text += chunk.text
            if chunk.input_tokens:
                input_tokens = chunk.input_tokens
            if chunk.output_tokens:
                output_tokens += chunk.output_tokens
    else:
        pprint(response)
        text = response.text
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
    total_cost = calculate_chat_cost(api_type, model, input_tokens, output_tokens)
    print(text)
    print(f'Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total cost: ${total_cost}')


if __name__ == "__main__":
    asyncio.run(main())
```
