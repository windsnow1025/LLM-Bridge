import asyncio
import logging
import os
from pprint import pprint

from dotenv import load_dotenv

from llm_bridge import *
from usage.function import function

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
        text="You are a helpful assistant.",
        files=[]
    ),
    Message(
        role=Role.User,
        text="What's this?",
        files=["https://www.windsnow1025.com/minio/windsnow/uploads/1/1740156158357-1740155993826-recording-1740155993821.webm"]
    )
]
model = "gemini-2.0-pro-exp-02-05"
api_type = "Gemini" # OpenAI / OpenAI-Azure / OpenAI-GitHub / Gemini / Claude / Grok
temperature = 0
stream = True

async def main():
    response = await function(api_keys, messages, model, api_type, temperature, stream)
    if stream:
        async for chunk in response:
            pprint(chunk)
    else:
        pprint(response)

if __name__ == "__main__":
    asyncio.run(main())
