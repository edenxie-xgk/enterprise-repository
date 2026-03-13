import logging
import os
import time

from core.settings import settings


def get_logger(
    name:str = 'agent',
    console_level:int = logging.INFO,
    file_level:int = logging.DEBUG,
    log_file = None,
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    os.makedirs(settings.log_dir, exist_ok=True)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(settings.log_format)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(settings.log_dir, f"{name}_{time.strftime("%Y%m%d")}.log")

    file_handler = logging.FileHandler(log_file,encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(settings.log_format)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
