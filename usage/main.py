import asyncio
import logging
import os
from pprint import pprint

from dotenv import load_dotenv

from llm_bridge import *
from llm_bridge.logic.model_prices import get_model_prices
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
            Content(type=ContentType.Text, data="Explain the concept of Occam's Razor and provide a simple, everyday example."),
        ]
    )
]
# See /llm_bridge/resources/model_prices.json for available models
model = "grok-3-fast-latest"
api_type = "Grok"
temperature = 0
stream = True


async def main():
    model_prices = get_model_prices()
    pprint(model_prices)
    response = await workflow(api_keys, messages, model, api_type, temperature, stream)
    if stream:
        async for chunk in response:
            pprint(chunk)
    else:
        pprint(response)


if __name__ == "__main__":
    asyncio.run(main())
