from typing import List


def normalize_distances(result_values: List[dict]) -> List[float]:
    min_value = 100
    max_value = 0

    for result in result_values:
        value = float(result["_distance"])
        if value > max_value:
            max_value = value
        if value < min_value:
            min_value = value

    normalized_values = []
    min_value = min(result["_distance"] for result in result_values)
    max_value = max(result["_distance"] for result in result_values)

    if max_value == min_value:
        # Avoid division by zero: Assign all normalized values to 0 (or any constant value like 1)
        normalized_values = [0 for _ in result_values]
    else:
        normalized_values = [(result["_distance"] - min_value) / (max_value - min_value) for result in
                             result_values]

    return normalized_values