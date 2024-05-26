from os import path
import logging
from pathlib import Path
logging.basicConfig(level=logging.DEBUG)
# ROOT_DIR = path.dirname(path.abspath(__file__))
#
# logging.debug("ROOT_DIR: ", ROOT_DIR)
#
# def get_absolute_path(path_from_root: str) -> str:
#     logging.debug("abspath: ", path.abspath(path.join(ROOT_DIR, path_from_root)))
#
#
#     return path.abspath(path.join(ROOT_DIR, path_from_root))
ROOT_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.DEBUG)
logging.debug("ROOT_DIR: %s", ROOT_DIR)

def get_absolute_path(path_from_root: str) -> str:
    absolute_path = ROOT_DIR / path_from_root
    logging.debug("abspath: %s", absolute_path.resolve())
    return str(absolute_path.resolve())