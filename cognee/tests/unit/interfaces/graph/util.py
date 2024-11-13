from datetime import datetime, timezone
from typing import Any, Dict


def run_test_against_ground_truth(
    test_target_item_name: str, test_target_item: Any, ground_truth_dict: Dict[str, Any]
):
    """Validates test target item attributes against ground truth values.

    Args:
        test_target_item_name: Name of the item being tested (for error messages)
        test_target_item: Object whose attributes are being validated
        ground_truth_dict: Dictionary containing expected values

    Raises:
        AssertionError: If any attribute doesn't match ground truth or if update timestamp is too old
    """
    for key, ground_truth in ground_truth_dict.items():
        if isinstance(ground_truth, dict):
            for key2, ground_truth2 in ground_truth.items():
                assert (
                    ground_truth2 == getattr(test_target_item, key)[key2]
                ), f"{test_target_item_name}/{key = }/{key2 = }: {ground_truth2 = } != {getattr(test_target_item, key)[key2] = }"
        else:
            assert ground_truth == getattr(
                test_target_item, key
            ), f"{test_target_item_name}/{key = }: {ground_truth = } != {getattr(test_target_item, key) = }"
    time_delta = datetime.now(timezone.utc) - getattr(test_target_item, "updated_at")

    assert time_delta.total_seconds() < 60, f"{ time_delta.total_seconds() = }"
