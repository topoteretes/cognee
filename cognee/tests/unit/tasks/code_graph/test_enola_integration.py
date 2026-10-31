"""Integration test that runs the real enola binary; skipped when not installed.

"Installed" is judged the way cognee itself resolves the binary — ENOLA_PATH,
the environment's scripts directory (the ``enola-cli`` wheel), or PATH.
"""

import pytest

from cognee.tasks.code_graph.enola import (
    EnolaNotInstalledError,
    find_enola_binary,
    parse_enola_snapshot,
    run_enola_generate,
)


def _enola_available() -> bool:
    try:
        find_enola_binary()
        return True
    except EnolaNotInstalledError:
        return False


pytestmark = pytest.mark.skipif(not _enola_available(), reason="enola binary is not installed")


@pytest.mark.asyncio
async def test_run_enola_generate_on_a_tiny_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("ENOLA_PATH", raising=False)

    repo_path = tmp_path / "tiny_repo"
    repo_path.mkdir()
    (repo_path / "main.go").write_text(
        'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("hello")\n}\n'
    )

    snapshot_dir = await run_enola_generate(repo_path)

    assert (snapshot_dir / "facts.jsonl").is_file()

    facts, _receipt = parse_enola_snapshot(snapshot_dir)
    assert isinstance(facts, list)
