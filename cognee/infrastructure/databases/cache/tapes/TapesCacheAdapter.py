"""Filesystem-backed cache adapter that mirrors QA writes to a running tapes
ingest server (https://github.com/papercomputeco/tapes).

The FS adapter remains the source of truth for session state. Every new QA
entry is additionally POSTed to `{tapes_ingest_url}/v1/ingest` as a provider-
shaped request/response turn, so the conversation becomes queryable through
tapes' Merkle DAG and semantic search surfaces. Tapes is append-only; QA
updates, deletes, agent-trace steps, and usage logs are not mirrored.
"""

import json
import time
import uuid

import httpx

from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter
from cognee.shared.logging_utils import get_logger

logger = get_logger("TapesCacheAdapter")


class TapesCacheAdapter(FSCacheAdapter):
    """FS adapter that also mirrors each new QA to tapes /v1/ingest."""

    def __init__(
        self,
        session_ttl_seconds: int | None = 604800,
        *,
        tapes_ingest_url: str = "http://localhost:8082",
        tapes_provider: str = "openai",
        tapes_agent_name: str = "cognee",
        tapes_model: str = "cognee-session",
        tapes_request_timeout: float = 5.0,
    ):
        super().__init__(session_ttl_seconds=session_ttl_seconds)
        self.tapes_ingest_url = tapes_ingest_url.rstrip("/")
        self.tapes_provider = tapes_provider
        self.tapes_agent_name = tapes_agent_name
        self.tapes_model = tapes_model
        self.tapes_request_timeout = tapes_request_timeout
        self._tapes_client: httpx.AsyncClient | None = None

    def _get_tapes_client(self) -> httpx.AsyncClient:
        if self._tapes_client is None:
            self._tapes_client = httpx.AsyncClient(timeout=self.tapes_request_timeout)
        return self._tapes_client

    def _build_openai_turn(self, *, question: str, context: str, answer: str) -> tuple[dict, dict]:
        messages: list[dict] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": question})

        request_body = {"model": self.tapes_model, "messages": messages}
        response_body = {
            "id": f"cognee-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.tapes_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
        }
        return request_body, response_body

    def _build_anthropic_turn(
        self, *, question: str, context: str, answer: str
    ) -> tuple[dict, dict]:
        request_body: dict = {
            "model": self.tapes_model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": question}],
        }
        if context:
            request_body["system"] = context
        response_body = {
            "id": f"msg_cognee_{uuid.uuid4()}",
            "type": "message",
            "role": "assistant",
            "model": self.tapes_model,
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": answer}],
        }
        return request_body, response_body

    def _build_ollama_turn(self, *, question: str, context: str, answer: str) -> tuple[dict, dict]:
        messages: list[dict] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": question})
        request_body = {"model": self.tapes_model, "messages": messages, "stream": False}
        response_body = {
            "model": self.tapes_model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": answer},
            "done": True,
            "done_reason": "stop",
        }
        return request_body, response_body

    def _build_provider_turn(
        self, *, question: str, context: str, answer: str
    ) -> tuple[dict, dict]:
        builders = {
            "openai": self._build_openai_turn,
            "anthropic": self._build_anthropic_turn,
            "ollama": self._build_ollama_turn,
        }
        builder = builders.get(self.tapes_provider, self._build_openai_turn)
        return builder(question=question, context=context, answer=answer)

    async def _mirror_to_tapes(self, *, question: str, context: str, answer: str) -> None:
        request_body, response_body = self._build_provider_turn(
            question=question, context=context, answer=answer
        )
        payload = {
            "provider": self.tapes_provider,
            "agent_name": self.tapes_agent_name,
            "request": json.loads(json.dumps(request_body)),
            "response": json.loads(json.dumps(response_body)),
        }
        try:
            client = self._get_tapes_client()
            resp = await client.post(f"{self.tapes_ingest_url}/v1/ingest", json=payload)
            if resp.status_code >= 400:
                logger.warning(
                    "Tapes ingest rejected turn: status=%s body=%s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as e:
            logger.warning("Tapes mirror failed, continuing with FS cache only: %s", e)

    async def create_qa_entry(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        qa_id: str | None = None,
        feedback_text: str | None = None,
        feedback_score: int | None = None,
        used_graph_element_ids: dict | None = None,
        memify_metadata: dict | None = None,
    ):
        await super().create_qa_entry(
            user_id,
            session_id,
            question,
            context,
            answer,
            qa_id,
            feedback_text,
            feedback_score,
            used_graph_element_ids=used_graph_element_ids,
            memify_metadata=memify_metadata,
        )
        await self._mirror_to_tapes(question=question, context=context, answer=answer)

    async def close(self):
        if self._tapes_client is not None:
            try:
                await self._tapes_client.aclose()
            except Exception as e:
                logger.debug("Error closing tapes HTTP client: %s", e)
            self._tapes_client = None
        await super().close()
