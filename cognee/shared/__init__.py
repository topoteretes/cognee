# Load the .env before anything in cognee.shared reads settings (data_models.py
# builds an LLM config at import). One resolver for the whole process, shared
# with cognee/__init__.py — see cognee.shared.env_file for the search order.
from cognee.shared.env_file import load_env_file

load_env_file()
