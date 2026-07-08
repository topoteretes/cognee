import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv, set_key

# Ensure we load the root .env file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(ENV_PATH)

class Settings(BaseSettings):
    # LLM Settings
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    
    # Embedding Settings
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIMENSIONS: str = "3072"
    
    # Database Settings
    RELATIONAL_DATABASE: str = "sqlite://./cognee.db"
    GRAPH_DATABASE: str = "networkx"
    VECTOR_DATABASE: str = "lancedb"
    
    # Dynamic settings for Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_LLM_MODEL: str = ""
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBEDDING_DIMENSIONS: str = "768"

    class Config:
        env_file = ENV_PATH
        extra = "ignore"

settings = Settings()

# Directory for file uploads
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def update_env_setting(key: str, value: str):
    """Updates a setting dynamically in the .env file and in-memory settings."""
    # Write to .env
    set_key(ENV_PATH, key, value)
    
    # Reload env
    load_dotenv(ENV_PATH, override=True)
    
    # Update settings object attributes
    if hasattr(settings, key):
        setattr(settings, key, value)
        
    # If LLM_PROVIDER changes, we update os.environ
    os.environ[key] = value
