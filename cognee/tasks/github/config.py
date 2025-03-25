"""
GitHub API configuration settings.
"""
import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class GitHubSettings(BaseSettings):
    """
    Settings for GitHub API access.
    
    This class loads configuration from environment variables and .env file.
    """
    GITHUB_API_TOKEN: Optional[str] = None
    github_api_base_url: str = "https://api.github.com"
    
    # Configure to load from .env file
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_github_settings():
    """
    Get GitHub API settings singleton.
    
    Returns:
        GitHubSettings: GitHub settings instance
    """
    return GitHubSettings() 