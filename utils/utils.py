import time


def is_chinese(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())