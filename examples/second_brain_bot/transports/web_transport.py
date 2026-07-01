"""Minimal web transport: a single FastAPI POST endpoint.

Thin by design. It normalizes one HTTP request to a Conversation and hands it
to the bot, then returns the reply. All memory, identity, and citation logic
lives in the shared layers, not here.

fastapi/pydantic are imported lazily so the package imports (and the no-key
tests) do not require them.

Note: this module intentionally does not use ``from __future__ import
annotations``. FastAPI must resolve the request-body model from the handler's
annotation, and a stringized annotation pointing at a closure-local Pydantic
model cannot be resolved, which would silently demote the body to a query param.
"""

from datetime import datetime, timezone
from typing import Optional

from ..adapter.interface import Conversation
from ..bot.router import Bot


def build_web_app(bot: Bot):
    """Build a FastAPI app exposing POST /message for the given bot."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Second Brain (web transport)")

    class InboundMessage(BaseModel):
        user: str  # stable per-user id for this web client
        text: str
        session: Optional[str] = None  # optional conversation id; defaults to the user

    @app.post("/message")
    async def message(payload: InboundMessage) -> dict:
        source = payload.session or payload.user
        ts = datetime.now(timezone.utc).isoformat()
        conversation = Conversation(
            transport="web",
            source=source,
            external_user=payload.user,
            msg_ref=f"web://{source}",
        )
        reply = await bot.handle(conversation, payload.text, ts)
        return {"reply": reply}

    return app
