"""Promote a selected persisted memory using pre-authorized dataset IDs.

Create the user/team datasets and grant read+share on the source and write on
its destination before calling this example. The team destination must already
be shared for reading with the tenant. No permissions are granted by promote().
"""

from uuid import UUID

import cognee
from cognee.modules.users.models import User


async def agent_to_user_to_team(
    memory_id: UUID,
    agent_dataset: UUID,
    user_dataset: UUID,
    team_dataset: UUID,
    agent: User,
    owner: User,
):
    # The agent selects a useful persisted lesson. Pending session entries must
    # first be persisted with improve(); promotion never copies the whole session.
    plan = await cognee.promote(
        memory_id,
        source_dataset_id=agent_dataset,
        target_dataset_id=user_dataset,
        level="user",
        reason="A verified lesson useful across this user's projects",
        user=agent,
        dry_run=True,
    )
    personal = await cognee.promote(
        plan.source_data_id,
        source_dataset_id=agent_dataset,
        target_dataset_id=user_dataset,
        level="user",
        reason="A verified lesson useful across this user's projects",
        user=agent,
    )
    # A separate selection/approval policy decides whether this personal lesson
    # belongs in team memory. The owner's permissions gate this second step.
    shared = await cognee.promote(
        personal.target_data_id,
        source_dataset_id=user_dataset,
        target_dataset_id=team_dataset,
        level="team",
        reason="Reviewed team practice",
        user=owner,
    )
    await cognee.cognify(datasets=[team_dataset], user=owner)
    return shared
