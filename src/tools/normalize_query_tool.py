def normalize_query(query: str) -> str:
    """清洗数据，规范化数据"""
    if not query:
        return ""

    # 去空格
    query = query.strip()

    # 简单统一大小写（英文）
    query = query.lower()

    return query