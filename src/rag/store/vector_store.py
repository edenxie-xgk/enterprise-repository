from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

from core.settings import settings
from src.models.embedding import embed_model
from src.models.llm import deepseek_llm



url = make_url(settings.database_string)
vector_store = PGVectorStore.from_params(
    database= url.database,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=settings.vector_table_name,  # 表名你可以随便起
    embed_dim=settings.embedding_dim,  # 你的向量维度
)

if __name__=="__main__":
    Settings.llm = deepseek_llm
    Settings.embed_model = embed_model
    query_engine = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model).as_query_engine(
        similarity_top_k=3,
    )

    print(query_engine.retrieve("新格林耐特网络有限公司"))
    print(query_engine.query("远航智能制造市值多少"))