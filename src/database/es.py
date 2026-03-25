from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from core.settings import settings

class ElasticsearchClient:
    def __init__(self, index_name:str):
        self.es = Elasticsearch(
            settings.elasticsearch_url,
            request_timeout = 30,
            verify_certs = True,
            ssl_show_warn = True
        )
        self.index = index_name

        body = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "metadata": {
                        "type": "object",
                    },
                    "node_id": {
                        "type": "text",
                    }
                }
            }
        }

        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index, body=body)

    def insert_many(self, docs:list[dict]):
        """
          批量插入文档
          Returns:
              bulk操作的结果
          """
        # 准备 bulk 操作的数据
        actions = []
        for doc in docs:
            action = {
                 "_index": self.index,
                "_source": {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "node_id": doc["node_id"]
                }
            }
            actions.append(action)

        # 执行 bulk 插入
        success, failed = bulk(self.es, actions, stats_only=True)

        print(f"成功插入: {success} 条，失败: {failed} 条")
        return {"success": success, "failed": failed}








