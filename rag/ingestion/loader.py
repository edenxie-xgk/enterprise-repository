from typing import Sequence

from core.custom_types import DocumentMetadata
from llama_index.core.schema import Document as LlamaDocument
from docx import Document
import re

def load_txt(path,metadata: DocumentMetadata)->Sequence[LlamaDocument]:
    """获取普通文本Document"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    metadata.section_title = ".".join(metadata.file_name.split(".")[:-1])
    return [
        LlamaDocument(
            text=text,
            metadata=metadata.dict()
        )
    ]

def load_docx(file_path: str, metadata: DocumentMetadata)->Sequence[LlamaDocument]:
    doc = Document(file_path)
    sections = []
    current_section = ""
    current_title = None
    for para in doc.paragraphs:
        text = para.text.strip()

        if not text:
            continue

        # 如果是标题
        if para.style.name.startswith("Heading"):

            if current_section:
                sections.append((current_title,current_section))

            current_title = text
            current_section = f"# {text}\n"
        else:
            current_section += text + "\n"

    if current_section:
        sections.append((current_title,current_section))

    # 解析表格
    for table in doc.tables:

        table_text = []

        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]

            table_text.append(" | ".join(row_data))

        sections.append((current_title,"\n".join(table_text)))

    documents = []

    for title,section in sections:
        metadata.section_title = title
        documents.append(
            LlamaDocument(
                text=section,
                metadata=metadata.dict()
            )
        )

    return documents




def load_markdown(path: str, metadata: DocumentMetadata) -> Sequence[LlamaDocument]:
    """获取markdown Document"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = []

    # 按 markdown 标题切分
    parts = re.split(r'(?=\n#{1,6} )', text)

    def extract_md_title(text: str) -> str | None:
        match = re.search(r'^(#{1,6})\s+(.*)', text, re.MULTILINE)
        if match:
            return match.group(2).strip()
        return None

    for part in parts:
        part = part.strip()
        if not part:
            continue
        metadata.section_title = extract_md_title(part)
        sections.append(
            LlamaDocument(
                text=part,
                metadata=metadata.dict()
            )
        )

    return sections


def load_file(path:str, metadata: DocumentMetadata)->Sequence[LlamaDocument]:
    if path.endswith(".txt"):
        return load_txt(path,metadata)
    elif path.endswith(".docx") or path.endswith(".doc"):
        return load_docx(path,metadata)
    elif path.endswith(".md"):
        return load_markdown(path,metadata)
    else:
        return []


