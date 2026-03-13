from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
import psycopg2

from core.settings import settings

# 1. 先创建数据库（如果还没有）
connection_string = settings.database_string
db_name = settings.database_name


# 2. 创建 vector store 实例
url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=settings.vector_table_name,  # 表名你可以随便起
    embed_dim=settings.embedding_dim,  # 你的向量维度
)


print(vector_store)