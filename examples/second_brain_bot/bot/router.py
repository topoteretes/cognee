"""The bot core: resolve identity, dispatch commands, then capture or recall.

Every transport funnels here: it normalizes a platform event to a Conversation
plus text and timestamp, and the router does the rest. Capture vs recall is a
small, predictable rule (a message ending in "?", or prefixed /ask or /recall,
is a recall; anything else is a note), so the demo stays deterministic and the
tests need no model.
"""

from __future__ import annotations

from dataclasses import replace

from ..adapter.interface import Answer, ChatMemoryAdapter, Conversation, Message
from .commands import CommandHandler
from .consent import ConsentStore
from ..identity.identity_store import IdentityStore
from ..identity.linking import LinkingService

_RECALL_PREFIXES = ("/ask ", "/recall ")
_NOTE_PREFIX = "/note "


def classify(text: str) -> tuple[str, str]:
    """Return ("recall" | "ingest", payload_text)."""
    stripped = text.strip()
    low = stripped.lower()
    for prefix in _RECALL_PREFIXES:
        if low.startswith(prefix):
            return "recall", stripped[len(prefix) :].strip()
    if low.startswith(_NOTE_PREFIX):
        return "ingest", stripped[len(_NOTE_PREFIX) :].strip()
    if stripped.endswith("?"):
        return "recall", stripped
    return "ingest", stripped


def render_reply(answer: Answer) -> str:
    """Format an Answer plus its citations into one transport-agnostic reply string."""
    if not answer.citations:
        return answer.text
    lines = [answer.text, "", "Sources:"]
    for citation in answer.citations:
        when = citation.timestamp.split("T")[0] if citation.timestamp else "unknown date"
        lines.append(
            f"  from your {citation.source_transport} note on {when}: "
            f"{citation.content} ({citation.source_ref})"
        )
    return "\n".join(lines)


class Bot:
    def __init__(
        self,
        adapter: ChatMemoryAdapter,
        identity_store: IdentityStore,
        linking: LinkingService,
        consent: ConsentStore,
    ) -> None:
        self._adapter = adapter
        self._identity = identity_store
        self._consent = consent
        self._commands = CommandHandler(adapter, identity_store, linking, consent)

    async def handle(self, conversation: Conversation, text: str, ts: str) -> str:
        """Process one inbound message and return the reply text.

        ``conversation`` arrives from a transport without ``canonical_user``;
        the router resolves it here.
        """
        canonical = self._identity.resolve(
            conversation.transport, conversation.external_user or conversation.source
        )
        conversation = replace(conversation, canonical_user=canonical)

        command_reply = await self._commands.handle(conversation, text)
        if command_reply is not None:
            return command_reply

        action, payload = classify(text)

        if action == "recall":
            answer = await self._adapter.answer(conversation, payload)
            return render_reply(answer)

        # ingest
        if not self._consent.is_allowed(canonical):
            return "Capture is paused. Send /optin to start saving notes again."
        if not payload:
            return "Send me a note to remember, or ask a question ending with ?."
        await self._adapter.ingest(
            conversation,
            Message(text=payload, ts=ts, deeplink=conversation.msg_ref),
        )
        return "Saved to your brain."
