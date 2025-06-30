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
model = "gemini-2.5-pro"
# model = "claude-sonnet-4-0"
# api_type = "OpenAI"
api_type = "Gemini-Free"
# api_type = "Gemini-Paid"
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
