"""Sync policies: how often a stored connection syncs, and whether it is paused (SDK-1434).

The policy table references the host's own ``integration_credentials`` table, so create
tables only where that model is registered: in the SDK, import
``cognee.modules.integrations.models``; Cloud defines its own.
"""

from .exceptions import SyncPausedError, SyncPolicyConflictError, SyncPolicyNotFoundError
from .next_due import next_due
from .policy import CUSTOM_HOURS, INTERVAL_PRESETS, SyncPolicy
from .store import (
    delete_sync_policy,
    get_caller_effective_sync_policy,
    get_effective_sync_policy,
    get_sync_pause,
    get_sync_policy,
    pause_sync_policy,
    policy_level,
    require_sync_not_paused,
    resume_sync_policy,
    save_sync_policy,
)
from .tick import run_due_syncs

__all__ = [
    "CUSTOM_HOURS",
    "INTERVAL_PRESETS",
    "SyncPausedError",
    "SyncPolicy",
    "SyncPolicyConflictError",
    "SyncPolicyNotFoundError",
    "delete_sync_policy",
    "get_caller_effective_sync_policy",
    "get_effective_sync_policy",
    "get_sync_pause",
    "get_sync_policy",
    "next_due",
    "pause_sync_policy",
    "policy_level",
    "require_sync_not_paused",
    "resume_sync_policy",
    "run_due_syncs",
    "save_sync_policy",
]
