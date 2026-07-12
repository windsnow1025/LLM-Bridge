import asyncio
import base64
import io
import logging
import sys
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from pprint import pprint

from usage.keys import api_keys
from usage.single.config import *
from usage.workflow import workflow

script_dir = Path(__file__).parent.resolve()

log_path = script_dir / "output.log"
log_file = log_path.open("w", encoding="utf-8")
sys.stdout = log_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=log_file
)

audio_path = script_dir / "output.wav"
audio_path.unlink(missing_ok=True)


def write_wav(audio_segments: list[str], audio_path: Path) -> None:
    audio_bytes = b"".join(base64.b64decode(segment) for segment in audio_segments)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_bytes)


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
        web_search,
        code_execution,
        structured_output_schema,
    )
    text = ""
    thought_text = ""
    code_text = ""
    code_output_text = ""
    files = []
    audio_segments = []

    if stream and isinstance(response, AsyncGenerator):
        async for chunk in response:
            pprint(chunk)
            if chunk.text:
                text += chunk.text
            if chunk.audio:
                audio_segments.append(chunk.audio)
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
    elif isinstance(response, ChatResponse):
        pprint(response)
        text = response.text
        thought_text = response.thought
        code_text = response.code
        code_output_text = response.code_output
        input_tokens = response.input_tokens or 0
        output_tokens = response.output_tokens or 0
        files = response.files
        if response.audio:
            audio_segments.append(response.audio)
    
    if audio_segments:
        write_wav(audio_segments, audio_path)

    total_cost = calculate_chat_cost(api_type, model, input_tokens, output_tokens)
    print(f"Thought:\n{thought_text}\n")
    print(f"Code:\n{code_text}\n")
    print(f"Code Output:\n{code_output_text}\n")
    print(f"Text:\n{text}\n")
    print(f"Files:\n{files}\n")
    print(f"Audio:\n{audio_path if audio_segments else None}\n")
    print(f'Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total cost: ${total_cost}')


if __name__ == "__main__":
    asyncio.run(main())
    log_file.close()
