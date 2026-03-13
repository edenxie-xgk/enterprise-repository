import time

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """文档元数据结构。"""
    file_name:str
    file_path:str
    file_type:str
    file_size:int
    create_time:str | int = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    user_id:str | int
    department:str


if  __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))