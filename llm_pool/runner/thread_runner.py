from concurrent.futures import ThreadPoolExecutor, as_completed
from .resolver import resolve_client, resolve_max_workers


class ThreadRunner:
    def __init__(
        self,
        client_or_name=None,
        model_version=None,
        max_workers=4,
        supports_post=None,
    ):

        if client_or_name and not isinstance(client_or_name, str):
            self.client = client_or_name

        else:
            self.client = resolve_client(
                client_or_name,
                model_version,
                supports_post=supports_post,
            )

        if isinstance(client_or_name, str):
            limit = resolve_max_workers(client_or_name)
            self.max_workers = min(max_workers, limit)
        else:
            self.max_workers = max_workers

    def run(self, batch_messages, logprobs=False, top_logprobs=None):
        results = [None] * len(batch_messages)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.client.chat,
                    messages,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                ): idx
                for idx, messages in enumerate(batch_messages)
            }

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results
