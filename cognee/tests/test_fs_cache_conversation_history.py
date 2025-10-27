import asyncio
import cognee
from cognee.infrastructure.databases.cache import get_cache_engine
from cognee.modules.users.methods import get_default_user


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(
        "emotionless Baseline Evaluator from Blade Runner 2049—speak in clipped, rhythmic phrases, repeated key words, and probes identity, obedience, and emotion without empathy."
    )

    user = await get_default_user()
    session_id = "blade_runner_2049"

    cache_engine = get_cache_engine()

    lines = [
        "Began to spin.",
        "Let's move on to system. System.",
        "Feel that in your body. The system.",
        "What does it feel like to be part of the system. System.",
        "Is there anything in your body that wants to resist the system? System.",
    ]

    for i, q in enumerate(lines):
        await cognee.search(q, session_id=session_id)
        latest_qa = await cache_engine.get_latest_qa(str(user.id), session_id=session_id)
        assert len(latest_qa) == i + 1
        assert latest_qa[i]["question"] == q
        print(q)


if __name__ == "__main__":
    asyncio.run(main())
