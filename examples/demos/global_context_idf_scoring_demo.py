from __future__ import annotations

import os


def format_weights(weights: dict[str, float]) -> str:
    return ", ".join(f"{entity_id}={weight:.3f}" for entity_id, weight in sorted(weights.items()))


def main() -> None:
    os.environ.setdefault("COGNEE_CLI_MODE", "true")
    os.environ.setdefault("COGNEE_LOG_FILE", "false")
    os.environ.setdefault("LOG_LEVEL", "ERROR")

    from cognee.tasks.memify.global_context_index.idf import (
        compute_idf_from_counts,
        entities_weight,
        entity_weight,
        weighted_jaccard,
    )

    chunk_count = 4
    entity_chunk_counts = {
        "alice": 1,
        "project-x": 2,
        "standup": 4,
    }
    idf_weights = compute_idf_from_counts(chunk_count, entity_chunk_counts)

    alice_project = {"alice", "project-x", "standup"}
    project_only = {"project-x", "standup"}
    missing_only = {"unknown-entity"}

    print("Global context IDF scoring demo")
    print(f"chunk_count: {chunk_count}")
    print(f"idf_weights: {format_weights(idf_weights)}")
    print(f"weight({sorted(alice_project)}): {entities_weight(alice_project, idf_weights):.3f}")
    print(f"weight({sorted(project_only)}): {entities_weight(project_only, idf_weights):.3f}")
    print(f"weight({sorted(missing_only)}): {entities_weight(missing_only, idf_weights):.3f}")
    print(f"missing entity weight: {entity_weight('unknown-entity', idf_weights):.3f}")
    print(f"ubiquitous entity weight: {entity_weight('standup', idf_weights):.3f}")
    print(
        "weighted_jaccard(alice_project, project_only): "
        f"{weighted_jaccard(alice_project, project_only, idf_weights):.3f}"
    )
    print(
        "weighted_jaccard(missing_only, {'standup'}): "
        f"{weighted_jaccard(missing_only, {'standup'}, idf_weights):.3f}"
    )


if __name__ == "__main__":
    main()
