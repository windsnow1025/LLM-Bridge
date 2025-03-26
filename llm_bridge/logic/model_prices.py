import importlib.resources
import json
from typing import TypedDict


class ModelPrice(TypedDict):
    apiType: str
    model: str
    input: float
    output: float


def load_json_file(package, filename):
    with importlib.resources.open_text(package, filename) as f:
        return json.load(f)


def get_model_prices() -> list[ModelPrice]:
    return load_json_file("llm_bridge.resources", "model_prices.json")
