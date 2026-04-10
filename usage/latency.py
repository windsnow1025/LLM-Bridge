import asyncio
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from llm_bridge import *
from usage.workflow import workflow

script_dir = Path(__file__).parent.resolve()
load_dotenv(script_dir / ".env")

TimeoutSeconds = 60
MaxRetries = 5

api_keys = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "AZURE_API_KEY": os.environ.get("AZURE_API_KEY"),
    "AZURE_API_BASE": os.environ.get("AZURE_API_BASE"),
    "GITHUB_API_KEY": os.environ.get("GITHUB_API_KEY"),
    "GOOGLE_AI_STUDIO_FREE_TIER_API_KEY": os.environ.get("GOOGLE_AI_STUDIO_FREE_TIER_API_KEY"),
    "GOOGLE_AI_STUDIO_API_KEY": os.environ.get("GOOGLE_AI_STUDIO_API_KEY"),
    "VERTEX_AI_API_KEY": os.environ.get("VERTEX_AI_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "XAI_API_KEY": os.environ.get("XAI_API_KEY"),
}


async def measure_first_chunk_latency(
        api_type: str,
        model: str,
) -> float:
    messages = [
        Message(
            role=Role.User,
            contents=[Content(type=ContentType.Text, data="Hello")]
        )
    ]

    start = time.perf_counter()
    response = await workflow(
        api_keys,
        messages,
        model,
        api_type,
        temperature=0,
        stream=True,
        thought=False,
        web_search=False,
        code_execution=False,
        structured_output_schema=None,
    )

    async for chunk in response:
        if chunk.error:
            raise RuntimeError(chunk.error)
        return time.perf_counter() - start

    raise RuntimeError("No response chunks received")


async def test_models(
        models: list[dict],
) -> list[tuple[str, str, float | None, str]]:
    results: list[tuple[str, str, float | None, str]] = []

    for price in models:
        api_type: str = price["apiType"]
        model: str = price["model"]

        latency: float | None = None
        error: str | None = None

        for attempt in range(1, MaxRetries + 1):
            try:
                latency = await asyncio.wait_for(
                    measure_first_chunk_latency(api_type, model),
                    timeout=TimeoutSeconds,
                )
                print(f"  [{attempt}/{MaxRetries}] OK      {api_type:28s} {model:35s} {latency:8.3f}s")
                break
            except asyncio.TimeoutError:
                error = f"TIMEOUT ({TimeoutSeconds}s)"
                print(f"  [{attempt}/{MaxRetries}] TIMEOUT {api_type:28s} {model:35s}")
            except Exception as e:
                error = str(e)
                print(f"  [{attempt}/{MaxRetries}] ERROR   {api_type:28s} {model:35s} - {e}")

        if latency is not None:
            status = "OK"
        else:
            status = f"FAILED: {error}"

        results.append((api_type, model, latency, status))

    return results


async def main():
    model_prices = get_model_prices()

    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for price in model_prices:
        groups[price["apiType"]].append(price)

    tasks = [test_models(models) for models in groups.values()]
    group_results = await asyncio.gather(*tasks)

    results: list[tuple[str, str, float | None, str]] = []
    for group_result in group_results:
        results.extend(group_result)

    print(f"\n{'=' * 95}")
    print(f"{'API Type':28s} {'Model':35s} {'Latency':>10s}  {'Status'}")
    print(f"{'-' * 95}")
    for api_type, model, latency, status in results:
        latency_str = f"{latency:.3f}s" if latency is not None else "N/A"
        print(f"{api_type:28s} {model:35s} {latency_str:>10s}  {status}")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    asyncio.run(main())
