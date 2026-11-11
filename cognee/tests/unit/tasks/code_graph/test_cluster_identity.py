"""A repository maps to the same nodes alone and in a cluster, in any order (enola 0.4.39).

The fixtures are real enola 0.4.39 snapshots of two one-file repositories,
``alpha`` and ``beta``, each holding a package directory of its own name:
``beta`` extracted alone, and the cluster in both orders. In a cluster enola
prefixes every member's ``file`` with its repo label (``beta/beta/code.py``)
and marks the fact ``repo_prefixed``; the fact id leaves the prefix out.
"""

import json
from pathlib import Path

import pytest

from cognee.tasks.code_graph.enola import EnolaSnapshotError, parse_enola_snapshot
from cognee.tasks.code_graph.extract_code_graph import map_facts_to_data_points
from cognee.tasks.code_graph.models import CodeModule, CodeSymbol

FIXTURES = Path(__file__).parent / "fixtures" / "cluster_identity"
SNAPSHOTS = ("solo_beta", "alpha_then_beta", "beta_then_alpha")


def _beta_nodes(snapshot: str) -> dict[str, dict]:
    facts, _receipt = parse_enola_snapshot(FIXTURES / snapshot)
    points = map_facts_to_data_points(facts, repo_scope="beta" if "then" in snapshot else None)
    return {
        str(point.id): {
            "type": type(point).__name__,
            "name": point.name,
            "file_path": point.file_path,
            "enola_id": point.enola_id,
            "fact_hash": point.fact_hash,
            "declarations": getattr(point, "declarations", None),
        }
        for point in points
        if isinstance(point, (CodeSymbol, CodeModule)) and point.repo == "beta"
    }


def test_a_member_maps_to_the_same_nodes_alone_and_in_either_cluster_order():
    solo, first, second = (_beta_nodes(snapshot) for snapshot in SNAPSHOTS)

    assert solo == first == second
    symbol = next(node for node in solo.values() if node["type"] == "CodeSymbol")
    assert (symbol["name"], symbol["file_path"]) == ("beta/code.send", "beta/code.py")
    assert symbol["enola_id"] == "d575c3768046ac0831a245760e96fabc"
    assert symbol["declarations"][0]["file"] == "beta/code.py"
    module = next(node for node in solo.values() if node["type"] == "CodeModule")
    assert (module["name"], module["file_path"]) == ("beta", "beta")


def test_the_prefix_comes_off_every_prefixed_fact_and_the_flag_with_it():
    facts, _receipt = parse_enola_snapshot(FIXTURES / "alpha_then_beta")

    assert all("repo_prefixed" not in fact for fact in facts)
    files = {(fact["repo"], fact["name"]): fact.get("file") for fact in facts if "file" in fact}
    assert files[("alpha", "alpha/code.send")] == "alpha/code.py"
    assert files[("beta", "beta")] == "beta"


def test_a_prefixed_fact_whose_file_lacks_its_label_is_refused(tmp_path):
    fact = {
        "id": "0" * 32,
        "kind": "symbol",
        "repo": "beta",
        "name": "beta/code.send",
        "file": "elsewhere/code.py",
        "repo_prefixed": True,
    }
    (tmp_path / "facts.jsonl").write_text(json.dumps(fact) + "\n")
    (tmp_path / "receipt.json").write_text(json.dumps({"snapshot_id": "sha256:x"}))

    with pytest.raises(EnolaSnapshotError, match="does not start with its repo label"):
        parse_enola_snapshot(tmp_path)
