import asyncio
import random
import time
from dataclasses import dataclass

from llm_bridge import *
from usage.keys import api_keys
from usage.batch.config import BackoffBase, Configs, MaxRetries, TestConfig, TimeoutSeconds
from usage.workflow import workflow


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
        config.messages,
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


async def test_model(
        api_type: str,
        model: str,
        config: TestConfig,
) -> TestResult:
    latency: float | None = None
    error: str | None = None

    for attempt in range(1, MaxRetries + 1):
        try:
            async with asyncio.timeout(TimeoutSeconds):
                latency = await measure_first_chunk_latency(api_type, model, config)
            print(f"  [{attempt}/{MaxRetries}] OK      {api_type:28s} {model:35s} [{config.name:10s}] {latency:8.3f}s")
            break
        except TimeoutError:
            error = f"TIMEOUT ({TimeoutSeconds}s)"
            print(f"  [{attempt}/{MaxRetries}] TIMEOUT {api_type:28s} {model:35s} [{config.name:10s}]")
        except Exception as e:
            error = str(e)
            print(f"  [{attempt}/{MaxRetries}] ERROR   {api_type:28s} {model:35s} [{config.name:10s}] - {e}")

        if attempt < MaxRetries:
            delay = random.uniform(0, BackoffBase * 2 ** attempt)
            await asyncio.sleep(delay)

    if latency is not None:
        status = "OK"
    else:
        status = f"FAILED: {error}"

    return TestResult(api_type, model, config.name, latency, status)


async def main():
    model_prices = get_model_prices()

    tasks = [
        test_model(price["apiType"], price["model"], config)
        for price in model_prices
        for config in Configs
    ]
    results = await asyncio.gather(*tasks)

    print(f"\n{'=' * 110}")
    print(f"{'API Type':28s} {'Model':35s} {'Config':12s} {'Latency':>10s}  {'Status'}")
    print(f"{'-' * 110}")
    for r in results:
        latency_str = f"{r.latency:.3f}s" if r.latency is not None else "N/A"
        print(f"{r.api_type:28s} {r.model:35s} {r.config_name:12s} {latency_str:>10s}  {r.status}")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    asyncio.run(main())
