import logging
from io import BytesIO

import httpcore
import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((httpx.ConnectError, httpcore.ConnectError)),
    reraise=True,
)
async def fetch_file_data(file_url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(file_url)
        except httpx.ConnectError as e:
            logging.exception(f"httpx.ConnectError while fetching {file_url}: {e}")
            raise
        except httpcore.ConnectError as e:
            logging.exception(f"httpcore.ConnectError while fetching {file_url}: {e}")
            raise
        except Exception as e:
            logging.exception(f"Unknown error while fetching {file_url}: {e}")
            raise e

        if response.status_code != 200:
            status_code = response.status_code
            text = response.text
            logging.error(f"Error {status_code}: {text}")
            raise HTTPException(status_code=status_code, detail=text)

        content_type = response.headers.get("Content-Type", "")
        return response.content, content_type
