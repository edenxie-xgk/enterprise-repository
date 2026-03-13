
from llama_index.core import Document

import os

from core.custom_types import DocumentMetadata


def load_normal_file(path):
    """获取普通文本Document"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text




def load_file(path:str):
    if  path.endswith(".txt") or path.endswith('.md') or path.endswith('.doc'):
        return load_normal_file(path)
    else:
        return None


