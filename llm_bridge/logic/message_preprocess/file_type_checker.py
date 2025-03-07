import mimetypes
import re

from llm_bridge.logic.file_fetch import fetch_file_data
from llm_bridge.logic.message_preprocess.code_file_extensions import code_file_extensions


def is_file_type_supported(file_name: str) -> bool:
    if file_name.endswith(('.doc', '.xls', '.ppt')):
        return False
    else:
        return True


async def get_file_type(file_url: str) -> tuple[str, str]:
    file_name = re.split(r'-(.+)', file_url.split('/')[-1])[1]
    file_extension = '.' + file_name.split('.')[-1].lower()
    if file_extension in code_file_extensions:
        return 'text', 'code'
    if file_extension in '.pdf':
        return 'text', 'pdf'
    if file_extension in ('.docx', '.doc'):
        return 'text', 'word'
    if file_extension in ('.xlsx', '.xls'):
        return 'text', 'excel'
    if file_extension in ('.pptx', '.ppt'):
        return 'text', 'ppt'
    if file_extension in '.mp3':
        return 'audio', 'mp3'
    if file_extension in '.wav':
        return 'audio', 'wav'
    if file_extension in '.webm': # currently unable to tell audio / video
        return 'audio', 'webm'

    mime_type, _ = mimetypes.guess_type(file_name)
    if not mime_type:
        _, mime_type = await fetch_file_data(file_url)

    if mime_type:
        return mime_type.split('/')[0], mime_type.split('/')[1]

    return 'unknown', 'unknown'
