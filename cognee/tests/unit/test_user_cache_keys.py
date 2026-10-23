from cognee.infrastructure.databases.cache.cache_db_interface import keys_owned_by_user


def test_keys_owned_by_user_keeps_other_users():
    keys = [
        "agent_sessions:user-a:one",
        "agent_sessions:user-ab:one",
        "agent_traces:user-a:one",
        "session_context:user-b:one",
        "usage_logs:user-a",
        "agent_sessions:user-a",
    ]

    assert keys_owned_by_user(keys, "user-a") == [
        "agent_sessions:user-a:one",
        "agent_traces:user-a:one",
    ]
