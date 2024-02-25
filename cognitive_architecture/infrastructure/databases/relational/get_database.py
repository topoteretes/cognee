from .general.adapter import RelationalDBAdapter

def get_database():
    return RelationalDBAdapter()
