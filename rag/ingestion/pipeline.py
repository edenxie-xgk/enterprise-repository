from llama_index.core.schema import Document

from core.custom_types import DocumentMetadata
from rag.ingestion.chunker import chunk_text


def pipeline(content,metadata:DocumentMetadata):
    doc = Document(
        text=content,
        metadata=metadata
    )
    try:
        node = None
        if metadata.file_type == 'txt':
            node = chunk_text(doc)

    except Exception as e:
        print(e)

    return node


