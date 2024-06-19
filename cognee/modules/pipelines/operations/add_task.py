from ..models import Pipeline, Task

def add_task(pipeline: Pipeline, task: Task):
    pipeline.tasks.append(task)
