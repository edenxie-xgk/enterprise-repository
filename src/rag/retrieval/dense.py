from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

from core.settings import settings


class DenseRetriever:

    def __init__(self, vector_store,storage_context, top_k: int = settings.retriever_top_k):
        self.vector_store = vector_store
        self.top_k = top_k
        self.storage_context = storage_context
        self.index = VectorStoreIndex.from_vector_store(
            storage_context=self.storage_context,
            vector_store=self.vector_store,
        )
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
            filters=None,
        )

    @staticmethod
    def _value_matches(actual, expected) -> bool:
        if isinstance(expected, (list, tuple, set)):
            return any(DenseRetriever._value_matches(actual, item) for item in expected)
        if actual == expected:
            return True
        return str(actual) == str(expected)

    @staticmethod
    def _build_vector_filters(filters=None):
        """构建向量存储的元数据过滤器

        将普通的字典格式的过滤条件转换为llama_index框架所需的MetadataFilters对象，
        用于向量检索时的元数据过滤。

        Args:
            filters: 过滤条件字典，格式为 {字段名: 期望值}。期望值可以是单个值，
                    如果是列表/元组/集合类型会被跳过(不处理多值条件)。
                    例如: {"category": "news", "year": 2023}

        Returns:
            MetadataFilters: 构造好的元数据过滤器对象，可直接用于向量检索。
                            如果无有效过滤条件则返回None。
        """
        if not filters:
            return None

        # 初始化元数据过滤器列表
        metadata_filters = []

        # 遍历过滤条件字典，构建单个MetadataFilter对象
        for key, expected in filters.items():
            # 跳过多值条件(列表/元组/集合)，这些条件由_matches_filters方法处理
            if isinstance(expected, (list, tuple, set)):
                continue

            # 构造单个字段的等值过滤条件
            metadata_filters.append(
                MetadataFilter(
                    key=key,          # 元数据字段名
                    value=expected,   # 期望匹配的值
                    operator=FilterOperator.EQ,  # 使用等值操作符
                )
            )

        # 如果没有生成有效过滤条件则返回None
        if not metadata_filters:
            return None

        # 将所有过滤条件组合成一个MetadataFilters对象，使用AND逻辑连接
        return MetadataFilters(
            filters=metadata_filters,  # 包含所有单个过滤条件
            condition=FilterCondition.AND,  # 使用AND逻辑连接多个条件
        )

    @staticmethod
    def _matches_filters(metadata: dict, filters=None) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            actual = metadata.get(key)
            if not DenseRetriever._value_matches(actual, expected):
                return False
        return True

    def run(self,search_queries:list[str],filters=None,top_k:int=None,score=settings.retrieval_min_score):
        all_results = []
        if top_k:
            self.retriever.similarity_top_k = top_k
        vector_filters = self._build_vector_filters(filters)
        for query in search_queries:
            if not query:
                continue
            self.retriever._filters = vector_filters
            results = self.retriever.retrieve(query)
            for node in results:
                if node.score < score:
                    continue
                # metadata过滤
                if not self._matches_filters(node.metadata, filters):
                    continue

                doc = {
                    "content": node.text,
                    "metadata": node.metadata,
                    "dense_score": node.score,
                    "node_id": node.node_id
                }
                all_results.append(doc)

        return all_results


