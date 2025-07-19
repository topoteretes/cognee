import os
from functools import wraps


def override_distributed(new_func):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, distributed=None, **kwargs):
            default_distributed_value = os.getenv("COGNEE_DISTRIBUTED", "False").lower() == "true"
            distributed = default_distributed_value if distributed is None else distributed

            if distributed:
                return await new_func(*args, **kwargs)
            else:
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator
