class WrongTaskOrderException(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TaskExecutionException(Exception):
    type: str
    message: str
    traceback: str

    def __init__(self, type: str, message: str, traceback: str):
        self.message = message
        self.type = type
        self.traceback = traceback
        super().__init__(message)
