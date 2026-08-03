import logging
import re
from typing import NoReturn

import httpx
from fastapi import HTTPException
from openai import APIStatusError


def raise_http_exception(e: Exception) -> NoReturn:
    if isinstance(e, httpx.HTTPStatusError):
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    if isinstance(e, APIStatusError):
        raise HTTPException(status_code=e.status_code, detail=e.message)
    logging.exception(e)
    match = re.search(r'\d{3}', str(e))
    if match:
        error_code = int(match.group(0))
    else:
        error_code = 500

    raise HTTPException(status_code=error_code, detail=str(e))
