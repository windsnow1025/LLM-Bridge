import asyncio
import logging
import os
import sys
from pprint import pprint

from dotenv import load_dotenv

from llm_bridge import *
from usage.workflow import workflow

output_file = open("./usage/output.log", "w", encoding="utf-8")
sys.stdout = output_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=output_file
)

load_dotenv(".env")

api_keys = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "AZURE_API_KEY": os.environ.get("AZURE_API_KEY"),
    "AZURE_API_BASE": os.environ.get("AZURE_API_BASE"),
    "GITHUB_API_KEY": os.environ.get("GITHUB_API_KEY"),
    "GEMINI_FREE_API_KEY": os.environ.get("GEMINI_FREE_API_KEY"),
    "GEMINI_PAID_API_KEY": os.environ.get("GEMINI_PAID_API_KEY"),
    "GEMINI_VERTEX_API_KEY": os.environ.get("GEMINI_VERTEX_API_KEY"),
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
            # Thinking
            Content(type=ContentType.Text, data="Explain the concept of Occam's Razor and provide a simple, everyday example."),

            # Web Search
            # Content(type=ContentType.Text, data="What's the weather in NYC today?"),

            # Image Generation
            # Content(type=ContentType.Text, data="Please generate an image of a cat."),

            # URL Context
            # Content(type=ContentType.Text, data="What is in https://www.windsnow1025.com/"),

            # Code Execution
            # Content(type=ContentType.Text, data="What is the sum of the first 50 prime numbers? Generate and run code for the calculation, and make sure you get all 50."),

            # File Output
            # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1758384216123-script.py"),
            # Content(type=ContentType.Text, data="Please implement a minimum example of Neural Network in `script.py`"),
        ]
    ),
    # Message(
    #     role=Role.User,
    #     contents=[
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746208707489-image.png"),
    #         Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746209841847-A%20Tutorial%20on%20Spectral%20Clustering.pdf"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746212253473-file_example_MP3_700KB.mp3"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1746212980820-file_example_MP4_480_1_5MG.mp4"),
    #         # Content(type=ContentType.File, data="https://www.windsnow1025.com/minio/windsnow/uploads/1/1753804900037-Calculus.docx"),
    #         Content(type=ContentType.Text, data="What's this?"),
    #     ]
    # ),
]
# See /llm_bridge/resources/model_prices.json for available models
# model = "gpt-5.1"
# model = "gpt-5"
# model = "gpt-5-pro"
# model = "gpt-4.1"
# model = "gemini-2.5-flash-image-preview"
# model = "gemini-flash-latest"
# model = "gemini-2.5-pro"
# model = "gemini-3-pro-preview"
# model = "grok-4-latest"
model = "claude-sonnet-4-5"
# model = "claude-opus-4-1"
# api_type = "OpenAI"
# api_type = "OpenAI-Azure"
# api_type = "OpenAI-GitHub"
# api_type = "Gemini-Free"
# api_type = "Gemini-Paid"
# api_type = "Gemini-Vertex"
api_type = "Claude"
# api_type = "Grok"
temperature = 0
stream = True
# stream = False


async def main():
    model_prices = get_model_prices()
    pprint(model_prices)

    input_tokens = 0
    output_tokens = 0
    response = await workflow(api_keys, messages, model, api_type, temperature, stream)
    text = ""
    thought = ""
    code = ""
    code_output = ""

    if stream:
        async for chunk in response:
            pprint(chunk)
            if chunk.text:
                text += chunk.text
            if chunk.thought:
                thought += chunk.thought
            if chunk.input_tokens:
                input_tokens = chunk.input_tokens
            if chunk.output_tokens:
                output_tokens += chunk.output_tokens
            if chunk.code:
                code += chunk.code
            if chunk.code_output:
                code_output += chunk.code_output
    else:
        pprint(response)
        text = response.text
        thought = response.thought
        code = response.code
        code_output = response.code_output
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
    total_cost = calculate_chat_cost(api_type, model, input_tokens, output_tokens)
    print(f"Thought:\n{thought}\n")
    print(f"Code:\n{code}\n")
    print(f"Code Output:\n{code_output}\n")
    print(f"Text:\n{text}")
    print(f'Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total cost: ${total_cost}')


if __name__ == "__main__":
    asyncio.run(main())
    output_file.close()
