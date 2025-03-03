import logging
from io import BytesIO

import httpx
from fastapi import HTTPException


async def fetch_file_data(file_url: str) -> tuple[BytesIO, str]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(file_url)
        except Exception as e:
            logging.exception(e)
            raise HTTPException(status_code=500, detail=str(e))

        if response.status_code != 200:
            status_code = response.status_code
            text = response.text
            logging.exception(f"Error {status_code}: {text}")
            raise HTTPException(status_code=status_code, detail=text)

        content_type = response.headers.get("Content-Type", "")
        return BytesIO(response.content), content_type
