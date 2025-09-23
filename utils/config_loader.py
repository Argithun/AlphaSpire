import os
import yaml
from pathlib import Path
from threading import Lock


class ConfigLoader:
    """
    A singleton configuration loader for the entire project.
    Priority: Environment Variables > config.yaml
    """
    _instance = None
    _config = {}
    _lock = Lock()

    def __new__(cls, config_path: str = "config.yaml"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._load_config(config_path)
            return cls._instance

    def _load_config(self, config_path: str):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        # 环境变量优先（如不在环境变量中则用 config.yaml 值）
        self._config = {
            "openai_base_url": os.getenv("OPENAI_BASE_URL", yaml_config.get("openai_base_url")),
            "openai_api_key": os.getenv("OPENAI_API_KEY", yaml_config.get("openai_api_key")),
            "openai_model_name": os.getenv("OPENAI_MODEL_NAME", yaml_config.get("openai_model_name")),

            "worldquant_account": os.getenv("WORLDQUANT_ACCOUNT", yaml_config.get("worldquant_account")),
            "worldquant_password": os.getenv("WORLDQUANT_PASSWORD", yaml_config.get("worldquant_password")),
            "worldquant_login_url": os.getenv("WORLDQUAN_LOGIN_URL", yaml_config.get("worldquant_login_url")),
            "worldquant_consultant_posts_url": os.getenv("WORLDQUANT_CONSULTANT_POSTS_URL",
                                                         yaml_config.get("worldquant_consultant_posts_url")),
        }

    @classmethod
    def get(cls, key: str, default=None):
        """
        获取配置值。
        使用方法：ConfigLoader.get("openai_api_key")
        """
        if cls._instance is None:
            cls()  # 初始化
        return cls._instance._config.get(key, default)

    @classmethod
    def all(cls) -> dict:
        """
        获取完整配置字典。
        """
        if cls._instance is None:
            cls()
        return cls._instance._config.copy()
