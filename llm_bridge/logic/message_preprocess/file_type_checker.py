import mimetypes
import os
import re
from pathlib import PurePosixPath

from llm_bridge.logic.file_fetch import fetch_file_data
from llm_bridge.logic.message_preprocess.code_file_extensions import code_file_extensions


def is_file_type_supported(file_name: str) -> bool:
    if file_name.endswith(('.doc', '.xls', '.ppt')):
        return False
    else:
        return True


async def get_file_type(file_url: str) -> tuple[str, str]:
    file_name: str = get_filename_without_timestamp(file_url)

    # Treat filenames without an extension as their own extension
    suffix: str = PurePosixPath(file_name).suffix.lower()
    file_extension: str = suffix if suffix else '.' + file_name.lower()

    if file_extension in code_file_extensions:
        return 'text', 'code'
    if file_extension == '.pdf':
        return 'text', 'pdf'
    if file_extension == '.docx':
        return 'text', 'word'
    if file_extension == '.xlsx':
        return 'text', 'excel'
    if file_extension == '.pptx':
        return 'text', 'powerpoint'
    # At present, unable to differentiate between audio and video files. Since web recordings are uploaded in webm format, treat webm files as audio by default.
    if file_extension == '.webm':
        return 'audio', 'webm'

    mime_type, _ = mimetypes.guess_type(file_name)
    if not mime_type:
        _, mime_type = await fetch_file_data(file_url)

    if mime_type:
        return mime_type.split('/')[0], mime_type.split('/')[1]

    return 'unknown', 'unknown'


def get_filename_without_timestamp(file_url: str) -> str:
    base_name = PurePosixPath(file_url).name
    match = re.search(r'-(.+)', base_name)
    if match:
        return match.group(1)
    else:
        return base_name
