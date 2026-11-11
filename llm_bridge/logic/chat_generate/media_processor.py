import base64

from llm_bridge.logic.file_fetch import fetch_file_data


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode('utf-8')


async def get_bytes_content_from_url(req_url: str) -> tuple[bytes, str]:
    file_data, media_type = await fetch_file_data(req_url)
    return file_data, media_type


async def get_base64_content_from_url(req_url: str) -> tuple[str, str]:
    file_data, media_type = await get_bytes_content_from_url(req_url)
    base64_data = bytes_to_base64(file_data)
    return base64_data, media_type
