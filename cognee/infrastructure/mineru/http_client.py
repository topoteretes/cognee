import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from PIL import Image

from cognee.config.mineru import MinerUSettings
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class MineruHTTPClientError(RuntimeError):
    """Base error raised by the MinerU HTTP client."""


class MineruHTTPResponseError(MineruHTTPClientError):
    """Raised when the MinerU server returns an unexpected response."""

    def __init__(self, message: str, *, status: Optional[int] = None, response_text: Optional[str] = None):
        detail = message
        if status is not None:
            detail = f"{message} (status={status})"
        if response_text:
            detail = f"{detail}: {response_text}"
        super().__init__(detail)
        self.status = status
        self.response_text = response_text


@dataclass(slots=True)
class _RequestResult:
    status_code: int
    json_data: Any
    text: str


class MineruHTTPClient:
    """
    Minimal HTTP client for interacting with a MinerU deployment that exposes an
    OpenAI-compatible chat completions API.
    """

    def __init__(self, settings: MinerUSettings):
        if not settings.server_url:
            raise ValueError("MinerU server URL must be configured.")

        self._settings = settings
        self._base_url = self._normalise_base_url(settings.server_url)
        self._headers = {"Content-Type": "application/json"}
        self._headers.update(settings.headers())
        self._model: Optional[str] = settings.model
        self._model_lock = asyncio.Lock()

    @staticmethod
    def _normalise_base_url(server_url: str) -> str:
        parsed = urlparse(server_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid MinerU server URL: {server_url}")

        normalised = server_url.rstrip("/")
        if normalised.endswith("/v1"):
            normalised = normalised[:-3]
        return normalised

    @property
    def _chat_url(self) -> str:
        return f"{self._base_url}/v1/chat/completions"

    @property
    def _models_url(self) -> str:
        return f"{self._base_url}/v1/models"

    async def extract_text(
        self,
        image_bytes: bytes,
        *,
        prompt: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> str:
        """
        Send the provided image to the MinerU HTTP endpoint and return the transcription.
        """

        if not image_bytes:
            raise ValueError("image_bytes must not be empty.")

        await self._ensure_model()

        png_bytes, image_format = self._ensure_png(image_bytes)
        data_url = self._to_data_url(png_bytes, image_format)
        prompt_text = prompt or self._settings.user_prompt

        messages: list[dict[str, Any]] = []
        if self._settings.system_prompt:
            messages.append({"role": "system", "content": self._settings.system_prompt})

        user_content: list[dict[str, Any]] = []
        if prompt_text:
            user_content.append({"type": "text", "text": prompt_text})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": self._settings.detail,
                },
            }
        )
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": self._settings.max_completion_tokens,
        }

        if source_name:
            logger.debug("Sending image to MinerU HTTP server", extra={"source": source_name})

        result = await self._request("POST", self._chat_url, json=payload)

        if result.status_code != httpx.codes.OK:
            raise MineruHTTPResponseError(
                "Unexpected status code from MinerU chat endpoint",
                status=result.status_code,
                response_text=result.text,
            )

        if not isinstance(result.json_data, dict):
            raise MineruHTTPResponseError(
                "MinerU response was not a JSON object",
                status=result.status_code,
                response_text=result.text,
            )

        if result.json_data.get("object") == "error":
            error_message = result.json_data.get("message") or result.text
            raise MineruHTTPResponseError(f"MinerU reported an error: {error_message}")

        choices = result.json_data.get("choices")
        if not (isinstance(choices, list) and choices):
            raise MineruHTTPResponseError("MinerU response did not include choices", response_text=result.text)

        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason not in (None, "stop", "length"):
            raise MineruHTTPResponseError(f"Unexpected finish reason: {finish_reason}", response_text=result.text)
        if finish_reason == "length":
            logger.warning("MinerU truncated response because of max token limit", extra={"source": source_name})

        message = choice.get("message")
        if not isinstance(message, dict):
            raise MineruHTTPResponseError("MinerU response missing message payload", response_text=result.text)

        content = message.get("content") or ""
        if not isinstance(content, str):
            raise MineruHTTPResponseError("MinerU message content is not a string", response_text=result.text)

        return content.strip()

    async def _ensure_model(self) -> None:
        if self._model:
            return

        async with self._model_lock:
            if self._model:
                return

            logger.debug("Fetching MinerU model list", extra={"url": self._models_url})
            result = await self._request("GET", self._models_url)

            if result.status_code != httpx.codes.OK:
                raise MineruHTTPResponseError(
                    "Failed to list models from MinerU",
                    status=result.status_code,
                    response_text=result.text,
                )

            data = result.json_data
            models = data.get("data") if isinstance(data, dict) else None

            if not isinstance(models, list) or not models:
                raise MineruHTTPResponseError(
                    "MinerU model list response did not include any models",
                    response_text=result.text,
                )

            if len(models) > 1:
                logger.info(
                    "MinerU returned multiple models, using the first entry.",
                    extra={"model_ids": [model.get("id") for model in models if isinstance(model, dict)]},
                )

            first = models[0]
            model_id = first.get("id") if isinstance(first, dict) else None
            if not isinstance(model_id, str) or not model_id:
                raise MineruHTTPResponseError(
                    "MinerU model response missing 'id' field",
                    response_text=result.text,
                )

            self._model = model_id

    async def _request(self, method: str, url: str, **kwargs: Any) -> _RequestResult:
        attempt = 0
        last_exception: Optional[Exception] = None
        total_retries = max(0, self._settings.max_retries)

        while attempt <= total_retries:
            try:
                async with httpx.AsyncClient(
                    timeout=self._settings.timeout_seconds,
                    follow_redirects=True,
                ) as client:
                    response = await client.request(method, url, headers=self._headers, **kwargs)
            except httpx.RequestError as exc:
                last_exception = exc
                logger.warning(
                    "MinerU HTTP request failed",
                    extra={"url": url, "attempt": attempt + 1, "error": str(exc)},
                )
            else:
                text = response.text
                try:
                    json_data = response.json()
                except ValueError:
                    json_data = None

                if response.status_code >= 500 and attempt < total_retries:
                    logger.warning(
                        "MinerU server returned %s, retrying",
                        response.status_code,
                        extra={"url": url, "attempt": attempt + 1},
                    )
                else:
                    return _RequestResult(response.status_code, json_data, text)

            attempt += 1
            if attempt <= total_retries:
                await asyncio.sleep(self._backoff_delay(attempt))

        raise MineruHTTPClientError(f"Failed to reach MinerU server after {total_retries + 1} attempts.") from (
            last_exception
        )

    def _backoff_delay(self, attempt: int) -> float:
        return self._settings.retry_backoff_factor * (2 ** (attempt - 1))

    @staticmethod
    def _ensure_png(image_bytes: bytes) -> tuple[bytes, str]:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue(), "png"

    @staticmethod
    def _to_data_url(image_bytes: bytes, image_format: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{image_format};base64,{encoded}"


__all__ = ["MineruHTTPClient", "MineruHTTPClientError", "MineruHTTPResponseError"]

