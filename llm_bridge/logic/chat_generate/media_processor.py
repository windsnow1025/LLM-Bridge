import base64

from llm_bridge.logic.file_fetch import fetch_file_data


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode('utf-8')


async def get_bytes_content_from_url(req_url: str) -> tuple[bytes, str]:
    file_data, media_type = await fetch_file_data(req_url)
    return file_data, media_type


async def get_base64_content_from_url(req_url: str) -> tuple[str, str]:
    media_data, media_type = await get_bytes_content_from_url(req_url)
    base64_media = bytes_to_base64(media_data)
    return base64_media, media_type


async def get_openai_image_content_from_url(req_img_url: str) -> str:
    base64_image, media_type = await get_base64_content_from_url(req_img_url)
    return f"data:{media_type};base64,{base64_image}"
