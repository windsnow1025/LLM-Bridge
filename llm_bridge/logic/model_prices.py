import json
from importlib.resources import files
from typing import TypedDict

from fastapi import HTTPException


class ModelPrice(TypedDict):
    apiType: str
    model: str
    input: float
    output: float


def load_json_file(package: str, filename: str):
    content = files(package).joinpath(filename).read_text(encoding="utf-8")
    return json.loads(content)


def get_model_prices() -> list[ModelPrice]:
    prices = load_json_file("llm_bridge.resources", "model_prices.json")
    for price in prices:
        price["input"] = float(price["input"])
        price["output"] = float(price["output"])
    return prices


def find_model_prices(api_type: str, model: str) -> ModelPrice | None:
    chat_prices = get_model_prices()
    for chat_price in chat_prices:
        if chat_price['apiType'] == api_type and chat_price['model'] == model:
            return chat_price
    return None


def calculate_chat_cost(
        api_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int
) -> float:
    model_pricing = find_model_prices(api_type, model)

    if model_pricing is None:
        raise HTTPException(status_code=400, detail="Invalid Model")

    input_cost = model_pricing["input"] * (input_tokens / 1_000_000)
    output_cost = model_pricing["output"] * (output_tokens / 1_000_000)
    total_cost = input_cost + output_cost

    return total_cost
