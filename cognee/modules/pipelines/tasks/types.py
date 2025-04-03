from typing import Any, AsyncGenerator, Callable, Coroutine, Generator, Union


TaskExecutable = Union[
    Callable[..., Any],
    Callable[..., Coroutine[Any, Any, Any]],
    AsyncGenerator[Any, Any],
    Generator[Any, Any, Any],
]
