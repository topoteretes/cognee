"""Scoping: chat → dataset + session mapping (pure logic, no cognee, no keys)."""

from cognee_telegram.scoping import resolve_scope


def test_dm_is_per_user():
    scope = resolve_scope(chat_type="private", chat_id=7, user_id=7)
    assert scope.dataset_name == "telegram_dm_7"
    assert scope.session_id == "telegram:dm:7"
    assert scope.is_private is True
    assert scope.thread_id is None


def test_group_is_per_chat():
    scope = resolve_scope(chat_type="supergroup", chat_id=-1001234567890, user_id=7)
    assert scope.dataset_name == "telegram_group_n1001234567890"
    assert scope.session_id == "telegram:group:-1001234567890"
    assert scope.is_private is False


def test_forum_topic_extends_chat():
    scope = resolve_scope(chat_type="supergroup", chat_id=-1001234567890, user_id=7, thread_id=55)
    assert scope.dataset_name == "telegram_group_n1001234567890_55"
    assert scope.session_id == "telegram:group:-1001234567890:55"
    assert scope.thread_id == 55


def test_per_user_in_group_splits_by_sender():
    a = resolve_scope(chat_type="group", chat_id=-100, user_id=1, per_user_in_group=True)
    b = resolve_scope(chat_type="group", chat_id=-100, user_id=2, per_user_in_group=True)
    assert a.dataset_name != b.dataset_name
    assert a.dataset_name.endswith("_user_1")
    assert b.dataset_name.endswith("_user_2")


def test_dataset_names_are_identifier_safe():
    # Negative ids must not leak a '-' into the dataset name, and positive vs
    # negative ids of the same magnitude must not collide.
    neg = resolve_scope(chat_type="group", chat_id=-100, user_id=1).dataset_name
    pos = resolve_scope(chat_type="group", chat_id=100, user_id=1).dataset_name
    assert "-" not in neg
    assert neg != pos
