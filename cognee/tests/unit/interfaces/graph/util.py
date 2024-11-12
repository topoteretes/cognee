from datetime import datetime, timezone


def run_test_against_ground_truth(
    test_target_item_name, test_target_item, ground_truth_dict
):
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

    assert time_delta.total_seconds() < 20, f"{ time_delta.total_seconds() = }"
