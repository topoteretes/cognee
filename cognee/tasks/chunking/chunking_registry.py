chunking_registry = {}

def register_chunking_function(name):
    def decorator(func):
        chunking_registry[name] = func
        return func
    return decorator

def get_chunking_function(name: str):
    return chunking_registry.get(name)