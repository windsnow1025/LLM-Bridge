from llm_bridge.logic.model_prices import get_model_prices


def test_get_model_prices_types():
    result = get_model_prices()
    for model_price in result:
        assert isinstance(model_price["apiType"], str)
        assert isinstance(model_price["model"], str)
        assert isinstance(model_price["input"], float)
        assert isinstance(model_price["output"], float)
