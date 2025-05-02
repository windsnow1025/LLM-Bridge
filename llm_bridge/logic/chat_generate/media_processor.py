import base64

from llm_bridge.logic.file_fetch import fetch_file_data
from llm_bridge.type.model_message import claude_message


async def get_raw_content_from_url(req_url: str) -> tuple[bytes, str]:
    file_data, media_type = await fetch_file_data(req_url)
    return file_data, media_type


async def get_openai_image_content_from_url(req_img_url: str) -> str:
    img_data, media_type = await fetch_file_data(req_img_url)
    base64_image = base64.b64encode(img_data).decode('utf-8')
    return f"data:{media_type};base64,{base64_image}"


async def get_claude_image_content_from_url(req_img_url: str) -> claude_message.ImageContent:
    img_data, media_type = await fetch_file_data(req_img_url)
    base64_image = base64.b64encode(img_data).decode('utf-8')
    image_source = claude_message.ImageSource(type="base64", media_type=media_type, data=base64_image)
    return claude_message.ImageContent(type="image", source=image_source)


async def get_openai_audio_content_from_url(req_audio_url: str) -> str:
    audio_data, media_type = await fetch_file_data(req_audio_url)
    encoded_string = base64.b64encode(audio_data).decode('utf-8')
    return encoded_string
