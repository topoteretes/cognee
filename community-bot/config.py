"""Centralised configuration for the community bot.

Reads from community-bot/.env (loaded via python-dotenv).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load community-bot/.env relative to this file (not CWD)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

# --- Cognee datasets -------------------------------------------------------
ORG_DATASET = "org_community"
AGENT_DATASET = "agent_support"


def user_dataset(discord_user_id: int | str) -> str:
    """Dataset name for a given Discord user."""
    return f"user_{discord_user_id}"


# --- External APIs ---------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

DOCS_REPO_OWNER = os.getenv("DOCS_REPO_OWNER", "topoteretes")
DOCS_REPO_NAME = os.getenv("DOCS_REPO_NAME", "cognee-docs")
DOCS_REPO_BRANCH = os.getenv("DOCS_REPO_BRANCH", "main")

CODE_REPO_OWNER = os.getenv("CODE_REPO_OWNER", "topoteretes")
CODE_REPO_NAME = os.getenv("CODE_REPO_NAME", "cognee")

MAX_ISSUES = int(os.getenv("MAX_ISSUES", "200"))
MAX_MDX_FILES = int(os.getenv("MAX_MDX_FILES", "100"))

# --- Discord (Day 3+) ------------------------------------------------------
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")
DISCORD_HELP_CHANNEL_ID = os.getenv("DISCORD_HELP_CHANNEL_ID", "")

# --- Anthropic (Day 2) -----------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
