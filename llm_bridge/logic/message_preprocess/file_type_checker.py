import mimetypes

from llm_bridge.logic.message_preprocess.code_file_extensions import code_file_extensions


def is_file_type_supported(file_name: str) -> bool:
    if file_name.endswith(('.doc', '.xls', '.ppt')):
        return False
    else:
        return True


def get_file_type(file_name: str) -> tuple[str, str]:
    if file_name.endswith(code_file_extensions):
        return 'text', 'code'
    if file_name.endswith('.pdf'):
        return 'text', 'pdf'
    if file_name.endswith(('.docx', '.doc')):
        return 'text', 'word'
    if file_name.endswith(('.xlsx', '.xls')):
        return 'text', 'excel'
    if file_name.endswith(('.pptx', '.ppt')):
        return 'text', 'ppt'
    if file_name.endswith('.mp3'):
        return 'audio', 'mp3'
    if file_name.endswith('.wav'):
        return 'audio', 'wav'
    if file_name.endswith('.webm'): # currently unable to tell audio / video
        return 'audio', 'webm'

    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type:
        return mime_type.split('/')[0], mime_type.split('/')[1]
    return 'unknown', 'unknown'
