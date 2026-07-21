from typing import Optional


def validate_batch_size(
    value: Optional[int], parameter_name: str, *, allow_none: bool = False
) -> None:
    """Require a positive integer for pipeline batch and concurrency limits."""
    if value is None and allow_none:
        return

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{parameter_name} must be a positive integer.")
