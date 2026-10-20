"""RememberResult reports the memory state the run left the dataset in (SDK-397)."""

from uuid import uuid4

from cognee.api.v1.remember.remember import RememberResult
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunErrored


def _result():
    return RememberResult(status="running", dataset_name="ds")


def test_the_memory_state_is_read_off_the_run_info():
    dataset_id, run_id = uuid4(), uuid4()
    completed = PipelineRunCompleted(
        pipeline_run_id=run_id, dataset_id=dataset_id, dataset_name="ds", memory_state="ready"
    )
    result = _result()
    result._resolve({dataset_id: completed})
    assert result.status == "completed"
    assert result.memory_state == "ready"
    assert result.to_dict()["memory_state"] == "ready"
    assert "memory_state='ready'" in repr(result)

    errored = PipelineRunErrored(
        pipeline_run_id=run_id,
        dataset_id=dataset_id,
        dataset_name="ds",
        payload="boom",
        memory_state="extraction_failed",
    )
    result = _result()
    result._resolve({dataset_id: errored})
    assert result.status == "errored"
    assert result.memory_state == "extraction_failed"


def test_a_run_info_without_a_state_leaves_it_unset_and_a_server_payload_sets_it():
    dataset_id = uuid4()
    result = _result()
    result._resolve(
        {
            dataset_id: PipelineRunCompleted(
                pipeline_run_id=uuid4(), dataset_id=dataset_id, dataset_name="ds"
            )
        }
    )
    assert result.memory_state is None
    assert "memory_state" not in result.to_dict()

    result._attach_improve_payload({"memory_state": "indexed"})
    assert result.memory_state == "indexed"
