"""Expose observed extraction/mapping gaps without claiming exhaustive analysis."""

from copy import deepcopy


def repository_coverage(repositories, repo=None):
    reports = []
    for repository in repositories.values():
        if repo is not None and repository.get("name") != repo:
            continue
        receipt = repository.get("last_receipt")
        delta = repository.get("last_delta")
        quality = receipt.get("quality") if isinstance(receipt, dict) else None
        mapping = delta.get("mapping") if isinstance(delta, dict) else None
        known = (
            isinstance(quality, dict)
            and isinstance(mapping, dict)
            and all(
                type(quality.get(key)) is int
                for key in (
                    "files_seen",
                    "files_parsed",
                    "files_skipped",
                    "dirs_skipped",
                    "parse_errors",
                )
            )
            and type(mapping.get("relations_skipped")) is int
        )
        gaps = isinstance(mapping, dict) and bool(mapping.get("relations_skipped"))
        if isinstance(quality, dict):
            gaps |= any(
                isinstance(quality.get(key), (int, float)) and quality[key] > 0
                for key in ("parse_errors", "files_skipped", "dirs_skipped")
            )
            census = quality.get("census")
            if isinstance(census, dict):
                gaps |= any(
                    isinstance(census.get(key), (int, float)) and census[key] > 0
                    for key in ("excluded_by_ignore", "excluded_by_kind", "skipped_with_cause")
                )
        reports.append(
            {
                "repo": repository.get("name"),
                "snapshot_id": repository.get("last_snapshot_id"),
                "mapping_version": repository.get("last_mapping_version"),
                "status": "partial" if gaps else "observed_no_gaps" if known else "unknown",
                "extraction": deepcopy(quality),
                "mapping": deepcopy(mapping),
                "git": deepcopy(receipt.get("git")) if isinstance(receipt, dict) else None,
                "skip_inventory": "extractor_report_only",
            }
        )
    return {
        "repositories": reports,
        "exhaustive": False,
        "interpretation": "An absent relationship means it was not observed; extraction and static analysis may omit relationships.",
    }
