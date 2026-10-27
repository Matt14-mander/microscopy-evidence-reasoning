"""HTTP adapter for a server exposing the chat/completions vision protocol."""

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request


class APIModel:
    def __init__(self, config, inference):
        self.config = config
        self.inference = inference
        self.endpoint = config["base_url"].rstrip("/") + "/chat/completions"
        self.key = os.environ.get(config.get("api_key_env", "VLM_API_KEY"))
        if not self.key and config.get("require_api_key", True):
            raise ValueError(f"Set environment variable {config.get('api_key_env', 'VLM_API_KEY')}")

    def payload(self, request):
        content = [{"type": "text", "text": request.prompt}]
        for image in request.images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        payload = {"model": self.config["name"], "messages": [{"role": "user", "content": content}],
                   "temperature": self.inference.get("temperature", 0),
                   "max_tokens": self.inference.get("max_new_tokens", 256)}
        if self.config.get("send_seed", False):
            payload["seed"] = request.seed
        return payload

    def generate(self, request):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = json.dumps(self.payload(request)).encode("utf-8")
        retries = self.config.get("max_retries", 2)
        for attempt in range(retries + 1):
            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.get("timeout", 120)) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("API returned non-text content")
                return content
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == retries:
                    # Do not persist response bodies that may contain sensitive headers.
                    raise RuntimeError(f"Model API HTTP {error.code}") from None
                error.close()
                time.sleep(min(2 ** attempt, 30))
            except urllib.error.URLError:
                if attempt == retries:
                    raise RuntimeError("Model API connection failed") from None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Unreachable retry state")
