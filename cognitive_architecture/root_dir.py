from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))

def get_absolute_path(path_from_root: str) -> str:
    return path.abspath(path.join(ROOT_DIR, path_from_root))
