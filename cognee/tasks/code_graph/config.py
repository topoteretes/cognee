"""Per-call options for the code graph routes (CODE and CODE_REPO)."""

from typing import TypedDict, cast


class CodeGraphConfig(TypedDict, total=False):
    """Options for code files and code repositories, passed as ``codegraph_config``.

    ``cognify()`` reads ``batch_size``.
    """

    # Nodes per snapshot batch and per graph write (node and edge writes alike).
    # Larger trades RAM for speed: the load holds that many DataPoints at once
    # and makes fewer writes. 2000 when absent.
    batch_size: int


CODEGRAPH_CONFIG_KEYS = frozenset(CodeGraphConfig.__annotations__)
_POSITIVE_INT_KEYS = ("batch_size",)


def validate_codegraph_config(codegraph_config: CodeGraphConfig | None) -> CodeGraphConfig:
    """Return the config as a dict, rejecting unknown keys and mistyped values.

    A key set to ``None`` counts as absent and is dropped. A count must be a
    real int: ``True`` is an int to Python, but never a batch size.
    """
    if codegraph_config is None:
        return {}
    if not isinstance(codegraph_config, dict):
        raise TypeError(f"codegraph_config must be a dict, got {type(codegraph_config).__name__}.")
    unknown = set(codegraph_config) - CODEGRAPH_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown codegraph_config keys: {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(CODEGRAPH_CONFIG_KEYS))}."
        )
    config = {key: value for key, value in codegraph_config.items() if value is not None}
    for key in _POSITIVE_INT_KEYS:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"codegraph_config[{key!r}] must be an int, got {type(value).__name__}."
            )
        if value < 1:
            raise ValueError(f"codegraph_config[{key!r}] must be >= 1, got {value}.")
    return cast(CodeGraphConfig, config)
