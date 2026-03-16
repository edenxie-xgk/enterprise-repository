from typing import Sequence

from langchain_core.documents import Document
from llama_index.core.node_parser import SentenceSplitter
from utils.logger_handler import logger
from core.settings import settings

def chunk_txt(doc: Sequence[Document]):
    """
    将文本切成 nodes
    """
    splitter = SentenceSplitter(
        chunk_size=settings.txt_chunk_size,
        chunk_overlap=settings.txt_chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(doc)
    return nodes

def chunk_docx(doc: Sequence[Document]):
    """
   将docx文档切成 nodes
   """
    splitter = SentenceSplitter(
        chunk_size=settings.docx_chunk_size,
        chunk_overlap=settings.docx_chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(doc)
    return nodes

def chunk_markdown(doc: Sequence[Document]):
    """
    将markdown切成 nodes
    """
    splitter = SentenceSplitter(
        chunk_size=settings.md_chunk_size,
        chunk_overlap=settings.md_chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(doc)
    return nodes


def chunk_file(doc: Sequence[Document]):
    if doc[0].metadata["file_type"] == "txt":
        return chunk_txt(doc)
    elif doc[0].metadata["file_type"] == "docx" or  doc[0].metadata["file_type"] == "doc":
        return chunk_docx(doc)
    elif doc[0].metadata["file_type"] == "md":
        return chunk_markdown(doc)
    else:
        return []