from langchain_core.documents import Document
from llama_index.core.node_parser import SentenceSplitter


def chunk_text(doc: Document):
    """
    将文本切成 nodes
    """
    splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents([doc])
    return nodes