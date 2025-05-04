import base64

from llm_bridge.logic.file_fetch import fetch_file_data


async def get_raw_content_from_url(req_url: str) -> tuple[bytes, str]:
    file_data, media_type = await fetch_file_data(req_url)
    return file_data, media_type


async def get_encoded_content_from_url(req_url: str) -> tuple[str, str]:
    img_data, media_type = await get_raw_content_from_url(req_url)
    base64_image = base64.b64encode(img_data).decode('utf-8')
    return base64_image, media_type


async def get_openai_image_content_from_url(req_img_url: str) -> str:
    base64_image, media_type = await get_encoded_content_from_url(req_img_url)
    return f"data:{media_type};base64,{base64_image}"
