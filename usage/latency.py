import asyncio
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from llm_bridge import *
from usage.workflow import workflow

script_dir = Path(__file__).parent.resolve()
load_dotenv(script_dir / ".env")

Timeout_Seconds = 60
Trial_Count = 3

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
        0,
        True,
        False,
        False,
        None,
    )

    async for _ in response:
        return time.perf_counter() - start

    return time.perf_counter() - start


async def main():
    model_prices = get_model_prices()

    results: list[tuple[str, str, float | None, float | None, str]] = []

    for price in model_prices:
        api_type: str = price["apiType"]
        model: str = price["model"]

        latencies: list[float] = []
        errors: set[str] = set()

        for trial in range(Trial_Count):
            try:
                latency = await asyncio.wait_for(
                    measure_first_chunk_latency(api_type, model),
                    timeout=Timeout_Seconds,
                )
                latencies.append(latency)
                print(f"  [{trial + 1}/{Trial_Count}] OK      {api_type:20s} {model:35s} {latency:8.3f}s")
            except asyncio.TimeoutError:
                errors.add(f"TIMEOUT ({Timeout_Seconds}s)")
                print(f"  [{trial + 1}/{Trial_Count}] TIMEOUT {api_type:20s} {model:35s}")
            except Exception as e:
                errors.add(str(e))
                print(f"  [{trial + 1}/{Trial_Count}] ERROR   {api_type:20s} {model:35s} - {e}")

        avg_latency = sum(latencies) / len(latencies) if latencies else None
        std_latency = math.sqrt(sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) if len(latencies) == Trial_Count else None
        if errors:
            error_str = " | ".join(sorted(errors))
            status = f"ERRORS ({len(latencies)}/{Trial_Count} OK): {error_str}" if avg_latency is not None else f"FAILED: {error_str}"
        else:
            status = "OK"

        results.append((api_type, model, avg_latency, std_latency, status))
        avg_str = f"{avg_latency:.3f}s" if avg_latency is not None else "N/A"
        std_str = f"{std_latency:.3f}s" if std_latency is not None else "N/A"
        print(f"  => AVG  {api_type:20s} {model:35s} {avg_str:>8s} ± {std_str:<8s}  {status}\n")

    print(f"{'=' * 110}")
    print(f"{'API Type':20s} {'Model':35s} {'Avg Latency':>12s} {'Std Dev':>10s}  {'Status'}")
    print(f"{'-' * 110}")
    for api_type, model, avg_latency, std_latency, status in results:
        avg_str = f"{avg_latency:.3f}s" if avg_latency is not None else "N/A"
        std_str = f"{std_latency:.3f}s" if std_latency is not None else "N/A"
        print(f"{api_type:20s} {model:35s} {avg_str:>12s} {std_str:>10s}  {status}")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    asyncio.run(main())
