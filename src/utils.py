# src/utils.py

import yaml
import os
import logging

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    从 YAML 配置文件中加载配置
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_file: str = None):
    """
    设置日志记录，若提供 log_file，则日志同时输出到该文件
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    if log_file:
        # 如果 log_file 路径不存在，则创建目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
