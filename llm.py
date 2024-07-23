from typing import List
from abc import ABC
from langchain_openai import ChatOpenAI


class BaseLLM(ABC):
    _api_key: str
    _support_models: List[str]
    _default_model: str
    _base_url: str
    _default_temperature: float = 0.5
    _default_max_tokens: int = 4096

    def __init__(self, *, api_key: str):
        self._api_key = api_key

    @property
    def support_models(self) -> List[str]:
        return self._support_models

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def default_temperature(self) -> float:
        return self._default_temperature

    @property
    def default_max_tokens(self) -> int:
        return self._default_max_tokens

    def get_chat_engine(self, *, model: str = None, temperature: float = None, max_tokens: int = None):
        model = model or self.default_model
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        return ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class DeepSeekLLM(BaseLLM):
    _support_models = ['deepseek-chat', 'deepseek-coder']
    _base_url = 'https://api.deepseek.com/v1'
    _default_model = 'deepseek-chat'
    _default_max_tokens = 4096


class OpenRouterLLM(BaseLLM):
    _support_models = [
        'openai/gpt-4o-mini', 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o',
        'nvidia/nemotron-4-340b-instruct', 'deepseek/deepseek-coder',
        'google/gemini-flash-1.5', 'deepseek/deepseek-chat',
        'liuhaotian/llava-yi-34b', 'qwen/qwen-110b-chat',
        'qwen/qwen-72b-chat', 'google/gemini-pro-1.5',
        'cohere/command-r-plus', 'anthropic/claude-3-haiku',
    ]
    _base_url = 'https://openrouter.ai/api/v1'
    _default_model = 'anthropic/claude-3.5-sonnet'
    _default_max_tokens = 16 * 1024


class TongYiLLM(BaseLLM):
    _support_models = ['qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-long']
    _default_model = 'qwen-turbo'
    _base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    _default_max_tokens: int = 2000
