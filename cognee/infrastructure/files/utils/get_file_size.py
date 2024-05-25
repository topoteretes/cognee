import os

def get_file_size(file_path: str):
    """Get the size of a file"""
    return os.path.getsize(file_path)
