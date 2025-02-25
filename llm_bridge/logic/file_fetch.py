from io import BytesIO

import httpx
from fastapi import HTTPException


async def fetch_file_data(file_url: str) -> tuple[BytesIO, str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(file_url)

        if response.status_code != 200:
            status_code = response.status_code
            text = response.text
            raise HTTPException(status_code=status_code, detail=text)

        content_type = response.headers.get("Content-Type")
        return BytesIO(response.content), content_type
