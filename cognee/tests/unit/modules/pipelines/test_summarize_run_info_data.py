"""``summarize_run_info_data`` is only called by the cognee-saas pod, and only with no data."""

from cognee.modules.pipelines.utils import summarize_run_info_data


def test_the_pod_gets_none_as_a_string():
    assert summarize_run_info_data(None) == "None"
