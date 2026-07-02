"""Proof bot: a runnable chat-memory bot in ~40 lines, no API keys required.

This is the reference consumer of the chat-memory adapter. It shows the whole
contract (consent, ``ingest`` (remember), ``answer`` (recall + citations), and
``forget`` (privacy)) end to end, using the in-memory backend so it runs with
no LLM, no database, and no keys. Swap ``InMemoryMemoryBackend()`` for
``CogneeMemoryBackend()`` and set ``LLM_API_KEY`` to get real knowledge-graph
memory with zero other changes.

Run it::

    python examples/integrations/chat_memory/console_bot.py
"""

import asyncio

from cognee.integrations.chat_memory import (
    ChatMemoryAdapter,
    Conversation,
    InMemoryMemoryBackend,
    Message,
    per_channel_scope,
)


class ConsoleBot:
    """A minimal 'platform' bot. Its only job is to translate events for the adapter."""

    def __init__(self) -> None:
        # A production bot would pass CogneeMemoryBackend() here instead.
        self.memory = ChatMemoryAdapter(
            scope=per_channel_scope,
            backend=InMemoryMemoryBackend(),
        )

    def _conversation(self, user: str) -> Conversation:
        # One "channel" for the demo; a real bot maps its platform event here.
        return Conversation(platform="console", workspace="demo", channel="general", user=user)

    async def on_message(self, user: str, text: str, ts: str) -> None:
        convo = self._conversation(user)
        stored = await self.memory.ingest(
            convo,
            Message(text=text, user=user, timestamp=ts, permalink=f"console://general/{ts}"),
        )
        flag = "remembered" if stored else "skipped (no consent)"
        print(f"  [{user}] {text!r} -> {flag}")

    async def on_question(self, user: str, question: str) -> None:
        answer = await self.memory.answer(self._conversation(user), question)
        print(f"\n  Q  {user} asks: {question}")
        if answer.is_empty:
            print("  A  I don't have anything on that yet.")
            return
        print(f"  A  {answer.text}")
        for c in answer.citations:
            print(f"       cite [{c.source}] by {c.user}: {c.permalink}")

    async def on_forget_me(self, user: str) -> None:
        result = await self.memory.forget(conversation=self._conversation(user), user=user)
        print(f"  forget(user={user!r}) -> removed {result['items_removed']} item(s)")


def banner(title: str) -> None:
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


async def main() -> None:
    bot = ConsoleBot()

    banner("1) Consent - a channel bot stays silent until users opt in")
    await bot.on_message("alice", "The launch is scheduled for Friday.", "1")
    bot.memory.set_consent("alice", True)
    bot.memory.set_consent("bob", True)

    banner("2) Ingest - remember messages (background, fire-and-forget)")
    await bot.on_message("alice", "The launch is scheduled for Friday.", "2")
    await bot.on_message("bob", "Marketing will announce the launch on Monday.", "3")
    await bot.on_message("alice", "Alice owns the release checklist.", "4")

    banner("3) Answer - recall with citations back to the source message")
    await bot.on_question("bob", "when is the launch?")
    await bot.on_question("bob", "who owns the release checklist?")

    banner("4) Forget me - privacy: wipe just one user's memory")
    await bot.on_forget_me("alice")
    # Alice's checklist fact is gone; Bob's marketing fact survives.
    await bot.on_question("bob", "who owns the release checklist?")
    await bot.on_question("bob", "when will marketing announce?")


if __name__ == "__main__":
    asyncio.run(main())
