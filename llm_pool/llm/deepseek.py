from .base import BaseLLMClient
from .config import get_llm_config


class DeepSeekClient(BaseLLMClient):
    def __init__(self,
                 api_key: str | None = None,
                 base_url: str | None = None,
                 model: str | None = None,
                 endpoint: str | None = None,
                 config_name: str | None = None,
                 supports_post: bool = True):
        cfg = get_llm_config(config_name or "deepseek")

        super().__init__(
            api_key = api_key or cfg["api_key"],
            base_url = base_url or cfg["base_url"],
            model = model or cfg["model"],
            endpoint = endpoint or cfg["endpoint"],
            verify_ssl = False,
            supports_post = supports_post,
        )

    def build_payload(self, messages):
        return {
            "model": self.model,
            "messages": messages,
        }
