"""Expose observed extraction/mapping gaps without claiming exhaustive analysis.

A repository is ``partial`` when the extractor reported parse errors, or the
mapping could not resolve a relation to a fact (anything but an external dependency).
"""

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
                for key in ("files_seen", "files_parsed", "files_skipped", "parse_errors")
            )
            and isinstance(mapping.get("by_reason"), dict)
        )
        # Observed gaps only. Files and directories the extractor ignores (its
        # ``files_skipped`` is the ignore-glob count), files a manifest claims and external
        # dependencies are expected and stay visible in the counters below; they do not make a
        # repository partial.
        gaps = False
        if isinstance(quality, dict):
            gaps |= isinstance(quality.get("parse_errors"), (int, float)) and (
                quality["parse_errors"] > 0
            )
        if isinstance(mapping, dict) and isinstance(mapping.get("by_reason"), dict):
            gaps |= any(
                reason != "external_dependency" and count
                for reason, count in mapping["by_reason"].items()
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
