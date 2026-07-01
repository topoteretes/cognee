"""Identity linking merges two front-ends onto one canonical user / one brain."""

import asyncio


def _extract_code(link_reply: str) -> str:
    for line in link_reply.splitlines():
        if line.startswith("Link code:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no link code in reply: {link_reply!r}")


def test_link_flow_merges_identities_and_shares_brain(harness):
    async def run():
        # First contact from each transport creates a separate brain.
        await harness.send("telegram", "alice_tg", "my passport number is 1234")
        await harness.send("web", "alice_web", "my gym locker code is 99")

        tg_canonical = harness.identity.resolve("telegram", "alice_tg")
        web_canonical = harness.identity.resolve("web", "alice_web")
        assert tg_canonical != web_canonical  # separate brains before linking

        # Link: issue a code on Telegram, redeem it on web.
        link_reply = await harness.send("telegram", "alice_tg", "/link")
        code = _extract_code(link_reply)
        redeem_reply = await harness.send("web", "alice_web", f"/link {code}")
        assert "Linked" in redeem_reply

        # Both external identities now resolve to one canonical user.
        assert harness.identity.resolve("telegram", "alice_tg") == harness.identity.resolve(
            "web", "alice_web"
        )

        # Web now shares Telegram's brain and can recall the Telegram note.
        reply = await harness.send("web", "alice_web", "what is my passport number?")
        assert "1234" in reply or "passport" in reply.lower()

    asyncio.run(run())
