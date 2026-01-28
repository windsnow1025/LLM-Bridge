import pytest

from llm_bridge.logic.message_preprocess.file_type_checker import get_file_type


@pytest.mark.asyncio
async def test_get_file_type_with_extension():
    file_type, sub_type = await get_file_type("https://example.com/1767243600000-markdown.md")
    assert file_type == "text"
    assert sub_type == "code"


@pytest.mark.asyncio
async def test_get_file_type_pdf():
    file_type, sub_type = await get_file_type("https://example.com/1767243600000-document.pdf")
    assert file_type == "text"
    assert sub_type == "pdf"


@pytest.mark.asyncio
async def test_get_file_type_dockerfile():
    file_type, sub_type = await get_file_type("https://example.com/1767243600000-Dockerfile")
    assert file_type == "text"
    assert sub_type == "code"
