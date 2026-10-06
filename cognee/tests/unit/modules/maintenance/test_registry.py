"""Pins the maintenance registry: names, order, triggering pipelines, the disable switch."""

import pytest

from cognee.modules.maintenance import (
    DEFAULT_JOBS,
    BaseMaintenanceJob,
    MaintenanceConfig,
    get_maintenance_config,
    job_names,
    validate_jobs,
    validate_jobs_disabled,
)

EXPECTED_ORDER = ["vector_compaction"]


def test_registry_order():
    assert job_names() == EXPECTED_ORDER


def test_every_job_is_a_maintenance_job_with_a_pipeline():
    for job in DEFAULT_JOBS:
        assert isinstance(job, BaseMaintenanceJob)
        assert job.pipelines, f"{job.name} is triggered by no pipeline"


def test_vector_compaction_follows_cognify_only():
    (job,) = [job for job in DEFAULT_JOBS if job.name == "vector_compaction"]
    assert job.pipelines == frozenset({"cognify_pipeline"})


def _job(name, pipelines=frozenset({"p"})):
    job = BaseMaintenanceJob()
    job.name = name
    job.pipelines = pipelines
    return job


def test_validate_jobs_rejects_bad_registries():
    with pytest.raises(ValueError, match="duplicate"):
        validate_jobs([_job("a"), _job("a")])
    with pytest.raises(ValueError, match="needs a name"):
        validate_jobs([_job("")])
    with pytest.raises(ValueError, match="no triggering pipeline"):
        validate_jobs([_job("a", frozenset())])


def test_jobs_disabled_is_parsed_and_validated(monkeypatch):
    assert MaintenanceConfig(jobs_disabled=" vector_compaction , ").jobs_disabled == [
        "vector_compaction"
    ]
    with pytest.raises(ValueError, match="valid names"):
        validate_jobs_disabled(["vector_compation"], DEFAULT_JOBS)

    monkeypatch.setenv("MAINTENANCE_JOBS_DISABLED", "no_such_job")
    get_maintenance_config.cache_clear()
    try:
        with pytest.raises(ValueError, match="no_such_job"):
            get_maintenance_config()
    finally:
        monkeypatch.delenv("MAINTENANCE_JOBS_DISABLED")
        get_maintenance_config.cache_clear()
