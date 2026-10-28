from typing import Any


def summarize_run_info_data(data: Any) -> str:
    """Return the "None" the cloud pod's stale-run recovery still writes into its closing row.

    cognee stores no ``run_info["data"]``. Delete this file and its export in ``utils/__init__.py``
    together, in the change that drops the pod's import.
    """
    return "None"
