"""Embeddable web chat widget powered by cognee memory.

Run a tiny FastAPI backend that any site can talk to via a single
``<script>`` tag. It doubles as an "ask our docs" assistant: seed a docs
corpus once and every visitor conversation can recall from it, with
inline **citations** to the source material.

The HTTP layer is intentionally thin — all memory behavior lives in
``ChatMemoryAdapter`` (``adapter.py``), which any other transport
(WhatsApp, Teams) can reuse unchanged.

Run it::

    uv run python examples/bots/web_widget/server.py

Then open http://localhost:8000 for the "ask our docs" demo page, or embed
the widget on your own site::

    <script src="http://localhost:8000/widget.js"
            data-site-id="acme" data-api="http://localhost:8000"></script>

Endpoints:
    POST /api/chat    -> {answer, citations, session_id}
    POST /api/forget  -> clear one conversation's memory
    GET  /            -> demo "ask our docs" page
    GET  /widget.js   -> the embeddable snippet
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Make the example importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from adapter import ChatMemoryAdapter

STATIC_DIR = Path(__file__).resolve().parent / "static"

# A couple of "docs" so the demo returns something without external setup.
# Replace these with your real docs/site content.
DEMO_DOCS: List[str] = [
    "Cognee is an open-source AI memory platform. It turns raw data into a "
    "knowledge graph that AI agents can recall from, replacing plain RAG "
    "with an Extract-Cognify-Load pipeline.",
    "You store data with remember(), build the graph, then query it with "
    "recall(). Each conversation is isolated by a session_id, so one user's "
    "chat never leaks into another's.",
    "Use forget() to delete memory for a conversation or dataset. Visitors "
    "can opt out of being remembered at any time.",
]

DEMO_SITE_ID = os.getenv("WIDGET_SITE_ID", "demo")

adapter = ChatMemoryAdapter(namespace="web", top_k=8)
app = FastAPI(title="cognee web chat widget")


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    visitor_id: str = "anonymous"
    site_id: str = DEMO_SITE_ID
    opt_in: bool = True
    use_docs: bool = True


class ForgetRequest(BaseModel):
    conversation_id: str
    visitor_id: str = "anonymous"
    site_id: str = DEMO_SITE_ID


@app.on_event("startup")
async def _seed_docs() -> None:
    """Seed the demo "ask our docs" corpus once, best-effort."""
    if os.getenv("WIDGET_SKIP_SEED") == "1":
        return
    try:
        await adapter.ingest_docs(site_id=DEMO_SITE_ID, documents=DEMO_DOCS)
    except Exception as error:  # noqa: BLE001 - demo should still boot
        print(f"[web_widget] docs seeding skipped: {error}")


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    conversation = adapter.conversation(
        site_id=req.site_id,
        visitor_id=req.visitor_id,
        conversation_id=req.conversation_id,
    )

    # A "/forget" message is a first-class command, not something to remember.
    if req.message.strip().lower() in ("/forget", "forget me"):
        await adapter.forget(conversation=conversation)
        return JSONResponse(
            {
                "answer": "Done — I've forgotten this conversation.",
                "citations": [],
                "session_id": conversation.session_id,
            }
        )

    answer = await adapter.answer(
        conversation=conversation,
        query=req.message,
        remember=req.opt_in,
        use_docs=req.use_docs,
    )
    return JSONResponse(answer.as_dict())


@app.post("/api/forget")
async def forget(req: ForgetRequest) -> JSONResponse:
    conversation = adapter.conversation(
        site_id=req.site_id,
        visitor_id=req.visitor_id,
        conversation_id=req.conversation_id,
    )
    cleared = await adapter.forget(conversation=conversation)
    return JSONResponse({"cleared": bool(cleared), "session_id": conversation.session_id})


@app.get("/")
async def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")


@app.get("/widget.js")
async def widget_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "widget.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
