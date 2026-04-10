import asyncio
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from llm_bridge import *
from usage.batch.config import Configs, MaxRetries, Messages, TestConfig, TimeoutSeconds
from usage.workflow import workflow

script_dir = Path(__file__).parent.resolve()
load_dotenv(script_dir.parent / ".env")

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


@dataclass
class TestResult:
    api_type: str
    model: str
    config_name: str
    latency: float | None
    status: str


async def measure_first_chunk_latency(
        api_type: str,
        model: str,
        config: TestConfig,
) -> float:
    start = time.perf_counter()
    response = await workflow(
        api_keys,
        Messages,
        model,
        api_type,
        temperature=config.temperature,
        stream=config.stream,
        thought=config.thought,
        web_search=config.web_search,
        code_execution=config.code_execution,
        structured_output_schema=config.structured_output_schema,
    )

    async for chunk in response:
        if chunk.error:
            raise RuntimeError(chunk.error)
        return time.perf_counter() - start

    raise RuntimeError("No response chunks received")


async def test_models(
        models: list[dict],
) -> list[TestResult]:
    results: list[TestResult] = []

    for price in models:
        api_type: str = price["apiType"]
        model: str = price["model"]

        for config in Configs:
            latency: float | None = None
            error: str | None = None

            for attempt in range(1, MaxRetries + 1):
                try:
                    latency = await asyncio.wait_for(
                        measure_first_chunk_latency(api_type, model, config),
                        timeout=TimeoutSeconds,
                    )
                    print(f"  [{attempt}/{MaxRetries}] OK      {api_type:28s} {model:35s} [{config.name:5s}] {latency:8.3f}s")
                    break
                except asyncio.TimeoutError:
                    error = f"TIMEOUT ({TimeoutSeconds}s)"
                    print(f"  [{attempt}/{MaxRetries}] TIMEOUT {api_type:28s} {model:35s} [{config.name:5s}]")
                except Exception as e:
                    error = str(e)
                    print(f"  [{attempt}/{MaxRetries}] ERROR   {api_type:28s} {model:35s} [{config.name:5s}] - {e}")

            if latency is not None:
                status = "OK"
            else:
                status = f"FAILED: {error}"

            results.append(TestResult(api_type, model, config.name, latency, status))

    return results


async def main():
    model_prices = get_model_prices()

    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for price in model_prices:
        groups[price["apiType"]].append(price)

    tasks = [test_models(models) for models in groups.values()]
    group_results = await asyncio.gather(*tasks)

    results: list[TestResult] = []
    for group_result in group_results:
        results.extend(group_result)

    print(f"\n{'=' * 110}")
    print(f"{'API Type':28s} {'Model':35s} {'Config':8s} {'Latency':>10s}  {'Status'}")
    print(f"{'-' * 110}")
    for r in results:
        latency_str = f"{r.latency:.3f}s" if r.latency is not None else "N/A"
        print(f"{r.api_type:28s} {r.model:35s} {r.config_name:8s} {latency_str:>10s}  {r.status}")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    asyncio.run(main())
