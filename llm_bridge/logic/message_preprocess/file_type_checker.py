import mimetypes
import os
import re

from llm_bridge.logic.file_fetch import fetch_file_data
from llm_bridge.logic.message_preprocess.code_file_extensions import code_file_extensions


def is_file_type_supported(file_name: str) -> bool:
    if file_name.endswith(('.doc', '.xls', '.ppt')):
        return False
    else:
        return True


async def get_file_type(file_url: str) -> tuple[str, str]:
    file_name = get_file_name(file_url)

    file_extension = '.' + file_name.split('.')[-1].lower() # Treat filenames without an extension as their own extension
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


# Without Timestamp
def get_file_name(file_url: str) -> str:
    base_name = os.path.basename(file_url)
    match = re.search(r'-(.+)', base_name)
    if match:
        return match.group(1)
    else:
        return base_name
