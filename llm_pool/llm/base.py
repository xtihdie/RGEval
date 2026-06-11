import time

import requests
import urllib3
from abc import ABC, abstractmethod

from .media import contains_image_inputs

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BaseLLMClient(ABC):
    supports_images = False

    def __init__(
        self,
        api_key,
        base_url,
        model,
        endpoint="/v1/chat/completions",
        verify_ssl=True,
        supports_post=True,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint = endpoint
        self.verify_ssl = verify_ssl
        self.supports_post = supports_post
        self._openai_client = None

    def _openai_base_url(self):
        base = self.base_url.rstrip("/")
        if self.endpoint and self.endpoint.startswith("/v1/") and not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def _get_openai_client(self):
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is required for non-POST providers."
                ) from exc
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self._openai_base_url(),
            )
        return self._openai_client

    @abstractmethod
    def build_payload(self, messages):
        pass

    def prepare_messages(self, messages):
        if not self.supports_images and contains_image_inputs(messages):
            raise ValueError("Image inputs are not supported for this provider.")
        return messages

    def chat(self, messages, logprobs=False, top_logprobs=None):
        prepared = self.prepare_messages(messages)
        payload = self.build_payload(prepared)
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs

        if self.supports_post:
            url = f"{self.base_url}{self.endpoint}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            max_retry = 20
            last_error = None
            for attempt in range(max_retry):
                try:
                    resp = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        verify=self.verify_ssl,
                        timeout=100,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        if logprobs:
                            return {
                                "content": content,
                                "logprobs": data["choices"][0].get("logprobs"),
                            }
                        return content

                    if resp.status_code == 429 or 500 <= resp.status_code < 600:
                        last_error = RuntimeError(
                            f"Request failed [{resp.status_code}]: {resp.text}"
                        )
                        time.sleep(min(60, 2 ** attempt))
                        continue

                    raise RuntimeError(
                        f"Request failed [{resp.status_code}]: {resp.text}"
                    )
                except requests.RequestException as exc:
                    last_error = exc
                    time.sleep(min(60, 2 ** attempt))
                    continue

            raise RuntimeError(f"POST provider request failed after {max_retry} retries: {url}") from last_error

        client = self._get_openai_client()
        response = client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        if logprobs:
            return {
                "content": content,
                "logprobs": getattr(response.choices[0], "logprobs", None),
            }
        return content
