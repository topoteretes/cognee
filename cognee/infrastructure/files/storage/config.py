from contextvars import ContextVar


file_storage_config = ContextVar("file_storage_config", default=None)
