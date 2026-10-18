"""A parent is set only through create_user, never by the public sign-up schema.

A parent sees its child's agents and sessions and is granted every dataset the
child creates, so a sign-up that could name a parent would let anyone push
datasets into another user's view.
"""

import uuid

from cognee.modules.users.models.User import InternalUserCreate, UserCreate


def test_public_sign_up_drops_a_parent_from_the_request():
    body = {"email": "puppet@example.com", "password": "pw", "parent_user_id": str(uuid.uuid4())}

    created = UserCreate(**body)

    assert "parent_user_id" not in UserCreate.model_fields
    # What fastapi-users' register route stores (create with safe=True).
    assert "parent_user_id" not in created.create_update_dict()


def test_create_user_keeps_the_parent_it_is_given():
    parent = uuid.uuid4()

    created = InternalUserCreate(email="agent@example.com", password="pw", parent_user_id=parent)

    # What create_user stores (create with safe=False).
    assert created.create_update_dict_superuser()["parent_user_id"] == parent
