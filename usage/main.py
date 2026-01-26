import asyncio
import logging
import os
import sys
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv

from usage.config import *
from usage.workflow import workflow

script_dir = Path(__file__).parent.resolve()

# Env
load_dotenv(script_dir / ".env")

# Logging Output File
output_path = script_dir / "output.log"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_file = output_path.open("w", encoding="utf-8")
sys.stdout = output_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=output_file
)

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


async def main():
    input_tokens = 0
    output_tokens = 0
    response = await workflow(
        api_keys,
        messages,
        model,
        api_type,
        temperature,
        stream,
        thought,
        code_execution,
        structured_output_schema,
    )
    text = ""
    thought_text = ""
    code_text = ""
    code_output_text = ""
    files = []

    if stream:
        async for chunk in response:
            pprint(chunk)
            if chunk.text:
                text += chunk.text
            if chunk.thought:
                thought_text += chunk.thought
            if chunk.input_tokens:
                input_tokens = chunk.input_tokens
            if chunk.output_tokens:
                output_tokens += chunk.output_tokens
            if chunk.code:
                code_text += chunk.code
            if chunk.code_output:
                code_output_text += chunk.code_output
            if chunk.files:
                files.extend(chunk.files)
    else:
        pprint(response)
        text = response.text
        thought_text = response.thought
        code_text = response.code
        code_output_text = response.code_output
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        files = response.files
    total_cost = calculate_chat_cost(api_type, model, input_tokens, output_tokens)
    print(f"Thought:\n{thought_text}\n")
    print(f"Code:\n{code_text}\n")
    print(f"Code Output:\n{code_output_text}\n")
    print(f"Text:\n{text}\n")
    print(f"Files:\n{files}\n")
    print(f'Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total cost: ${total_cost}')


if __name__ == "__main__":
    asyncio.run(main())
    output_file.close()
