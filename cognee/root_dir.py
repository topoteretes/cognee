from os import path
import logging
ROOT_DIR = path.dirname(path.abspath(__file__))

logging.debug("ROOT_DIR: ", ROOT_DIR)

def get_absolute_path(path_from_root: str) -> str:
    logging.debug("abspath: ", path.abspath(path.join(ROOT_DIR, path_from_root)))


    return path.abspath(path.join(ROOT_DIR, path_from_root))
