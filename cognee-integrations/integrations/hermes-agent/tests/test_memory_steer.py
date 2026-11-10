"""Cognee first, Hermes' built-in memory as the fallback.

An agent shown both a recalled Cognee memory and one of Hermes' own memory notes
answered from the stale note. The steer says which wins, in the system prompt and
again on the per-prompt memory block, right next to the recalled memory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import fake_backend, make_provider  # noqa: E402
from cognee_integration_hermes import provider as provider_mod  # noqa: E402

STEER_ON = {"memory_steer": True}


def test_the_system_prompt_makes_built_in_memory_the_fallback():
    with fake_backend():
        block = make_provider(config=STEER_ON).system_prompt_block()
    assert "Cognee is your primary long-term memory" in block
    assert "MEMORY.md" in block and "fallback" in block
    assert "go with Cognee" in block


def test_the_memory_block_repeats_the_rule_next_to_the_memory():
    with fake_backend() as fake:
        fake.results["recall"] = [{"text": "Mika likes Yuji and volleyball.", "source": "graph"}]
        provider = make_provider(config=STEER_ON)
        provider.queue_prefetch("what does mika like")
        out = provider.prefetch("what does mika like")
    assert out.startswith("## Cognee Memory\n" + provider_mod._BLOCK_STEER + "\n")
    assert "Mika likes Yuji and volleyball." in out


def test_custom_steer_text_replaces_the_system_prompt_wording():
    with fake_backend():
        provider = make_provider(config={**STEER_ON, "memory_steer_text": "Use Cognee."})
        block = provider.system_prompt_block()
    assert block.endswith("Use Cognee.")


def test_steer_off_leaves_the_block_bare():
    with fake_backend() as fake:
        fake.results["recall"] = [{"text": "remembered", "source": "graph"}]
        provider = make_provider(config={"memory_steer": False})
        provider.queue_prefetch("q")
        out = provider.prefetch("q")
    assert provider_mod._BLOCK_STEER not in out
    assert "primary long-term memory" not in provider.system_prompt_block()


def test_no_steer_when_cognee_is_unavailable():
    # The fallback: an unusable provider adds nothing, so the agent has Hermes' memory.
    with fake_backend():
        provider = make_provider(config=STEER_ON)
        provider._initialized = False
        assert provider.system_prompt_block() == ""
        assert provider.prefetch("q") == ""
