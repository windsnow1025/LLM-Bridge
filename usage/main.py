import asyncio
import os
from pprint import pprint

from dotenv import load_dotenv

from llm_bridge import *
from usage.function import function

load_dotenv(".env")

api_keys = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "AZURE_API_KEY": os.environ.get("AZURE_API_KEY"),
    "AZURE_API_BASE": os.environ.get("AZURE_API_BASE"),
    "GITHUB_API_KEY": os.environ.get("GITHUB_API_KEY"),
    "GOOGLE_AI_STUDIO_API_KEY": os.environ.get("GOOGLE_AI_STUDIO_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
}

messages = [
    Message(role=Role.System, text="You are a helpful assistant.", file_urls=[]),
    Message(role=Role.User, text="What's today's news?", file_urls=[])
]
model = "gemini-2.0-flash-exp"
api_type = "Gemini"
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
