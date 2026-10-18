from contextvars import ContextVar

# Annotated, not just defaulted: without the parameter the type is inferred
# from ``default=None`` alone, so every reader looks like it is testing None
# for truth and the real dict a dataset context sets is invisible to a type
# checker (ty reported the guard in get_storage_config as always falsy).
file_storage_config: ContextVar[dict[str, str] | None] = ContextVar(
    "file_storage_config", default=None
)
