from .base import BaseLLMClient
from .media import normalize_messages_with_images
from .config import get_llm_config


class QwenVLClient(BaseLLMClient):
    supports_images = True

    def __init__(self,
                 api_key: str | None = None,
                 base_url: str | None = None,
                 model: str | None = None,
                 endpoint: str | None = None,
                 config_name: str | None = None,
                 supports_post: bool = True):
        cfg = get_llm_config(config_name or "qwen-vl")

        super().__init__(
            api_key = api_key or cfg["api_key"],
            base_url = base_url or cfg["base_url"],
            model = model or cfg["model"],
            endpoint = endpoint or cfg["endpoint"],
            verify_ssl = False,
            supports_post = supports_post,
        )

    def prepare_messages(self, messages):
        return normalize_messages_with_images(messages)

    def build_payload(self, messages):
        return {
            "model": self.model,
            "messages": messages,
        }
