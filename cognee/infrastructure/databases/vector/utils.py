from typing import List


def normalize_distances(result_values: List[dict]) -> List[float]:
    """
    Normalize distances in the provided result values.

    This function takes a list of dictionaries containing distance values and normalizes
    these distances to a range of 0 to 1. If all distances are the same, it assigns all
    normalized values to 0 to avoid division by zero.

    Parameters:
    -----------

        - result_values (List[dict]): A list of dictionaries, each containing a '_distance'
          key with a numeric value to be normalized.

    Returns:
    --------

        - List[float]: A list of normalized float values corresponding to the distances in
          the input list.
    """
    min_value = min(result["_distance"] for result in result_values)
    max_value = max(result["_distance"] for result in result_values)

    if max_value == min_value:
        # Avoid division by zero: Assign all normalized values to 0 (or any constant value like 1)
        normalized_values = [0 for _ in result_values]
    else:
        normalized_values = [
            (result["_distance"] - min_value) / (max_value - min_value) for result in result_values
        ]

    return normalized_values
