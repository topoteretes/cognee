"""/forget me wipes the brain across every transport and drops the identity links."""

import asyncio


def test_forget_me_wipes_across_transports(harness):
    async def run():
        canonical = harness.identity.resolve("telegram", "alice_tg")
        harness.identity.link("web", "alice_web", canonical)

        await harness.send("telegram", "alice_tg", "my wifi password is hunter2")

        # Recallable from the other transport before forgetting.
        pre = await harness.send("web", "alice_web", "what is my wifi password?")
        assert "hunter2" in pre

        forget_reply = await harness.send("web", "alice_web", "/forget me")
        assert "wiped" in forget_reply.lower()

        # Identity links for that brain are gone (asserted before any further
        # message re-creates an auto link).
        assert harness.identity.identities_for(canonical) == []

        # Recall returns nothing on BOTH transports now.
        tg_reply = await harness.send("telegram", "alice_tg", "what is my wifi password?")
        web_reply = await harness.send("web", "alice_web", "what is my wifi password?")
        assert "hunter2" not in tg_reply
        assert "hunter2" not in web_reply
        assert "do not have anything" in tg_reply.lower()
        assert "do not have anything" in web_reply.lower()

    asyncio.run(run())
