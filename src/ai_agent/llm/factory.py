from typing import Literal

from src.ai_agent.config import LLMConfig, config
from src.ai_agent.llm.base import BaseLLM
from src.ai_agent.llm.deepseek import DeepSeekLLM

ProviderType = Literal["deepseek"]


def create_llm(
    provider: ProviderType = "deepseek", llm_config: LLMConfig = None
) -> BaseLLM:
    if llm_config is None:
        llm_config = config.llm

    if provider == "deepseek":
        return DeepSeekLLM(llm_config)

    raise ValueError(f"Unsupported LLM provider: {provider}")
