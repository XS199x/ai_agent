import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# 用 config.py 自身位置推算项目根目录下的 .env 绝对路径
# src/ai_agent/config.py → 上 3 级 → 项目根目录
# 无论从哪个目录启动，都能正确找到 .env
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_env_file = os.path.join(_project_root, ".env")


def _load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and value:
                os.environ.setdefault(key, value)


# 关键：在创建任何 Settings 之前，先把 .env 的内容加载到 os.environ
# 这样所有嵌套的 BaseSettings 类都能通过 env_prefix 读取到变量
_load_env_file(_env_file)


class LLMConfig(BaseSettings):
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    stream_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_")


class AppConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    model_config = SettingsConfigDict(env_prefix="APP_")


class AgentConfig(BaseSettings):
    max_iterations: int = 10
    tool_call_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="AGENT_")


class Config(BaseSettings):
    llm: LLMConfig = LLMConfig()
    app: AppConfig = AppConfig()
    agent: AgentConfig = AgentConfig()

    model_config = SettingsConfigDict(env_file=_env_file, extra="ignore")


config = Config()
