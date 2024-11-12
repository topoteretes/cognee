from unittest.mock import Mock, create_autospec
from uuid import UUID, uuid4
from typing import Optional, List
from cognee.modules.users.models import User


def get_mock_user(
    user_id: Optional[UUID] = None,
    groups: Optional[List["Group"]] = None,
    **additional_attributes
) -> Mock:
    """
    Creates a mock User instance with configurable attributes.

    Args:
        user_id: Optional UUID for the user. Generates random UUID if not provided.
        groups: Optional list of group mocks to associate with the user.
        **additional_attributes: Any additional attributes to set on the mock user.

    Returns:
        Mock: A configured mock User instance.
    """
    # Generate a random UUID if none provided
    user_id = user_id or uuid4()

    # Create base mock
    mock_user = create_autospec(User, instance=True)

    # Configure basic attributes
    mock_user.id = user_id
    mock_user.__tablename__ = "users"
    mock_user.groups = groups or []

    # Set polymorphic identity
    mock_user.__mapper_args__ = {"polymorphic_identity": "user"}

    # Add any additional attributes
    for attr_name, attr_value in additional_attributes.items():
        setattr(mock_user, attr_name, attr_value)

    return mock_user
