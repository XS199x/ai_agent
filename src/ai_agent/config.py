from pydantic_settings import BaseSettings, SettingsConfigDict


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


config = Config()