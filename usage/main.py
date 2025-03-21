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
    "GOOGLE_AI_STUDIO_API_KEY": os.environ.get("GOOGLE_AI_STUDIO_API_KEY"),
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
            Content(type=ContentType.Text, data="What's this document about?"),
            Content(type=ContentType.File, data="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"),
            Content(type=ContentType.Text, data="And what about this image?  Can you describe it?"),
            Content(type=ContentType.File, data="https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg"),
            Content(type=ContentType.Text, data="Thanks for the info!"),
        ]
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
