from zhipuai import ZhipuAI

from core.settings import settings

Zhipu_client = ZhipuAI(api_key=settings.zhipuai_api_key)

def web_search_tool(query):
    return Zhipu_client.web_search.web_search( search_engine="search_std",
       search_query=query,
       count=10,  # 返回结果的条数，范围1-50，默认10
       content_size="high"  # 控制网页摘要的字数，默认medium)
    )




if __name__ == "__main__":
    res = web_search_tool(query="一句话解释python是什么？")
    for item in res.search_result:
        print(item.title)
        print(item)
        print("***" * 50)