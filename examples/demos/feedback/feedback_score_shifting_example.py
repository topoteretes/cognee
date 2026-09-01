import asyncio
from pathlib import Path

import cognee
from cognee import SearchType
from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import ERROR, setup_logging

cache_config = get_cache_config()
if not cache_config.caching or cache_config.cache_backend != "fs":
    raise CogneeConfigurationError(
        "feedback_score_shifting_example requires caching=True and CACHE_BACKEND=fs."
    )

DATA_DIR = Path(__file__).parent / "feedback_score_shifting_example_data"
TEXT_1 = (DATA_DIR / "german_car_manufacturers.txt").read_text()

TEXT_2 = (DATA_DIR / "us_tech_companies.txt").read_text()


async def main():
    await cognee.forget(everything=True)

    await cognee.remember([TEXT_1, TEXT_2], self_improvement=False)

    user = await get_default_user()
    session_id = "feedback_influence_minimal_demo"

    print("Step 1: Ask cars-specific question and give positive feedback (5).")
    await cognee.recall(
        query_text="Which German car manufacturers are described and what are they known for?",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        session_id=session_id,
    )
    qa_cars = (await cognee.session.get_session(session_id=session_id, user=user, last_n=1))[0]
    await cognee.session.add_feedback(
        session_id=session_id,
        qa_id=qa_cars.qa_id,
        feedback_score=5,
        feedback_text="Cars-focused context is exactly what I want.",
        user=user,
    )
    print("  Added feedback score=5 for cars context.\n")

    print("Step 2: Ask companies-specific question and give negative feedback (1).")
    await cognee.recall(
        query_text="Which technology companies are described and what are their products?",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        session_id=session_id,
    )
    qa_companies = (await cognee.session.get_session(session_id=session_id, user=user, last_n=1))[0]
    await cognee.session.add_feedback(
        session_id=session_id,
        qa_id=qa_companies.qa_id,
        feedback_score=1,
        feedback_text="Companies-focused context is less useful for this goal.",
        user=user,
    )
    print("  Added feedback score=1 for companies context.\n")

    print("Step 3: Apply feedback into graph feedback_weight values (memify).")
    await apply_feedback_weights_pipeline(user=user, session_ids=[session_id], alpha=0.9)
    print("  Feedback weights applied.\n")

    print("Step 4: Ask one neutral query while sweeping beta.")
    print(
        "  As beta increases, ranking should shift toward positively-rated context (companies focused on car manufacturers)."
        " 1 means only feedback score is taken into account nothing else.\n"
    )
    final_query = "List the companies in the context"
    for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        answer = await cognee.recall(
            query_text=final_query,
            query_type=SearchType.GRAPH_COMPLETION,
            user=user,
            feedback_influence=beta,
        )
        print(f"\n--- beta = {beta:.1f} ({beta * 100:.0f}% feedback influence) ---")
        print(str(answer))


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
