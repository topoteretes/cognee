"""Command handling: /link, /forget, /optin, /optout, /help.

Commands are the control plane of the bot. Anything that is not a command is a
note to remember or a question to recall (decided by the router). Each handler
returns a reply string; the handler returns None when the text is not a
command, so the router can fall through to capture/recall.
"""

from __future__ import annotations

from typing import Optional

from ..adapter.interface import ChatMemoryAdapter, Conversation
from ..bot.consent import ConsentStore
from ..identity.identity_store import IdentityStore
from ..identity.linking import LinkingService

_HELP_TEXT = (
    "Second brain commands:\n"
    "  send any note to remember it\n"
    "  ask a question (end with ?) to recall\n"
    "  /link            issue a code to connect another app to this brain\n"
    "  /link <code>     enter a code from your other app to share one brain\n"
    "  /forget me       wipe your whole brain across every app\n"
    "  /optout          pause capturing new notes\n"
    "  /optin           resume capturing new notes\n"
    "  /help            show this message"
)


class CommandHandler:
    def __init__(
        self,
        adapter: ChatMemoryAdapter,
        identity_store: IdentityStore,
        linking: LinkingService,
        consent: ConsentStore,
    ) -> None:
        self._adapter = adapter
        self._identity = identity_store
        self._linking = linking
        self._consent = consent

    async def handle(self, conversation: Conversation, text: str) -> Optional[str]:
        """Return a reply if text is a command, else None.

        ``conversation.canonical_user`` must already be resolved by the router.
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        canonical = conversation.canonical_user

        if command == "/help":
            return _HELP_TEXT

        if command == "/link":
            return self._handle_link(conversation, canonical, argument)

        if command == "/forget":
            if argument.lower() in ("me", "everything", "all"):
                return await self._handle_forget_me(canonical)
            return "To wipe your whole brain across every app, send: /forget me"

        if command == "/optout":
            self._consent.opt_out(canonical)
            return "Capture paused. I will not save new notes until you send /optin."

        if command == "/optin":
            self._consent.opt_in(canonical)
            return "Capture on. Send me notes and I will remember them."

        return f"Unknown command {command}. Send /help to see what I can do."

    def _handle_link(self, conversation: Conversation, canonical: str, argument: str) -> str:
        if not argument:
            code = self._linking.issue_code(canonical)
            return (
                f"Link code: {code}\n"
                "Open your other app and send: /link " + code + "\n"
                "Both apps will then share this brain."
            )
        linked = self._linking.redeem_code(
            argument, conversation.transport, conversation.external_user or conversation.source
        )
        if linked is None:
            return "That link code is invalid or expired. Issue a fresh one with /link."
        return "Linked. This app now shares one brain with your other app."

    async def _handle_forget_me(self, canonical: str) -> str:
        # Wipe the whole brain, then drop identity links so no app re-attaches,
        # and clear any consent preference.
        await self._adapter.forget(canonical)
        self._identity.unlink_all(canonical)
        self._consent.reset(canonical)
        return "Done. Your brain is wiped across every connected app, and the links are removed."
