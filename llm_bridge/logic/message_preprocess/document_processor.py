def apply_docxlatex_fix():
    import docxlatex.parser.ommlparser
    from docxlatex.parser.utils import qn

    original_class = docxlatex.parser.ommlparser.OMMLParser

    def safe_parse_d(self, root):
        bracket_map = {
            "(": "\\left(",
            ")": "\\right)",
            "[": "\\left[",
            "]": "\\right]",
            "{": "\\left{",
            "}": "\\right}",
            "〈": "\\left\\langle",
            "〉": "\\right\\rangle",
            "⟨": "\\left\\langle", # patch
            "⟩": "\\right\\rangle", # patch
            "⌊": "\\left\\lfloor",
            "⌋": "\\right\\rfloor",
            "⌈": "\\left\\lceil",
            "⌉": "\\right\\rceil",
            "|": "\\left|",
            "‖": "\\left\\|",
            "⟦": "[\\![",
            "⟧": "]\\!]",
        }
        }
        text = ""
        start_bracket = "("
        end_bracket = ")"
        seperator = "|"
        for child in root:
            if child.tag == qn("m:dPr"):
                for child2 in child:
                    if child2.tag == qn("m:begChr"):
                        start_bracket = child2.attrib.get(qn("m:val"))
                    if child2.tag == qn("m:endChr"):
                        end_bracket = child2.attrib.get(qn("m:val"))
                    if child2.tag == qn("m:sepChr"):
                        seperator = child2.attrib.get(qn("m:val"))
        for child in root:
            if child.tag == qn("m:e"):
                if text:
                    text += seperator
                text += self.parse(child)
        end_bracket_replacements = {
            "|": "\\right|",
            "‖": "\\right\\|",
            "[": "\\right[",
        }
        start_bracket_replacements = {
            "]": "\\left]",
        }
        if start_bracket:
            if start_bracket in start_bracket_replacements:
                text = start_bracket_replacements[start_bracket] + " " + text
            else:
                text = bracket_map[start_bracket] + " " + text
        if end_bracket:
            if end_bracket in end_bracket_replacements:
                text += " " + end_bracket_replacements[end_bracket]
            else:
                text += " " + bracket_map[end_bracket]
        return text

    setattr(original_class, 'parse_d', safe_parse_d)

    if hasattr(original_class, 'parsers'):
        original_class.parsers[qn("m:d")] = safe_parse_d

apply_docxlatex_fix()


import logging
from io import BytesIO

import fitz
import openpyxl
from docxlatex import Document as DocxLatexDocument
from fastapi import HTTPException
from pptx import Presentation

from llm_bridge.logic.file_fetch import fetch_file_data
from llm_bridge.logic.message_preprocess import file_type_checker


async def extract_text_from_file(file_url: str) -> str:
    file_content, content_type = await fetch_file_data(file_url)

    if file_type_checker.is_file_type_supported(file_url) is False:
        raise HTTPException(status_code=415,
                            detail=f"legacy filetypes ('.doc', '.xls', '.ppt') are not supported - {file_url}")

    file_type, sub_type = await file_type_checker.get_file_type(file_url)

    try:
        if sub_type == "code":
            return extract_text_from_code(file_content)
        if sub_type == "pdf":
            return extract_text_from_pdf(file_content)
        if sub_type == "word":
            return extract_text_from_word(file_content)
        if sub_type == "excel":
            return extract_text_from_excel(file_content)
        if sub_type == "powerpoint":
            return extract_text_from_ppt(file_content)
    except Exception as e:
        logging.exception(e)
        raise HTTPException(
            status_code=415,
            detail=f"Error during text extraction - {file_url} ({file_type}/{sub_type}) - {str(e)}"
        )

    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError as e:
        message = f"Unicode Decode Error during text extraction - {file_url} ({file_type}/{sub_type}) - {str(e)}"
        logging.error(message)
        return message


def extract_text_from_code(file_content: bytes) -> str:
    return file_content.decode('utf-8')


def extract_text_from_pdf(file_content: bytes) -> str:
    text = ""
    with fitz.open(stream=file_content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_word(file_content: bytes) -> str:
    with BytesIO(file_content) as f:
        doc = DocxLatexDocument(f)
        text_with_equations = doc.get_text()
        return text_with_equations


def extract_text_from_excel(file_content: bytes) -> str:
    text = ""
    workbook = openpyxl.load_workbook(BytesIO(file_content), data_only=True)
    for sheet in workbook:
        text += f"Sheet: {sheet.title}\n"
        for row in sheet.iter_rows(values_only=True):
            text += "\t".join([str(cell) if cell is not None else "" for cell in row]) + "\n"
    return text


def extract_text_from_ppt(file_content: bytes) -> str:
    text = ""
    presentation = Presentation(BytesIO(file_content))
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text
