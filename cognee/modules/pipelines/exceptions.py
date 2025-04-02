class WrongTaskOrderException(Exception):
    message: str


class TaskExecutionException(Exception):
    type: str
    message: str
    traceback: str
