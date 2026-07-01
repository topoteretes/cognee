import asyncio
import os
import cognee
from cognee.shared.logging_utils import setup_logging, INFO
from cognee.tasks.notetaker.normalize import normalize_transcript
from cognee.modules.retrieval.notetaker_templates import (
    NotetakerActionItemRetriever,
    NotetakerTemporalDeltaRetriever
)

async def main():
    # 1. Provide mock transcript turns
    turns = [
        ("Alice", "Let's release v1 today.", "2026-06-10 10:00"),
        ("Bob", "I'll handle the deployment.", "2026-06-10 10:05"),
        ("Alice", "Great, let's also plan v2 for next week.", "2026-06-10 10:10")
    ]
    
    # 2. Normalize transcript
    normalized_text = normalize_transcript(
        turns=turns,
        meeting_id="standup_1",
        permalink="https://example.com/standup/1"
    )
    
    print(f"Normalized Text:\n{normalized_text}\n")
    
    # 3. Add to cognee dataset (series_id)
    series_id = "engineering_standups"
    await cognee.add(normalized_text, dataset_name=series_id)
    
    # 4. Cognify the dataset with temporal_cognify=True
    print("Cognifying transcript with temporal awareness...")
    await cognee.cognify(
        datasets=[series_id],
        temporal_cognify=True
        # run_in_background=False for the example to wait for completion
    )
    print("Cognify complete!\n")
    
    # 5. Recall action items
    print("Recalling action items...")
    retriever = NotetakerActionItemRetriever()
    action_items = await retriever.get_completion("What are the action items from the meeting?")
    print(f"Action Items:\n{action_items}\n")
    
    # 6. Recall temporal delta
    print("Recalling temporal delta...")
    delta_retriever = NotetakerTemporalDeltaRetriever()
    delta = await delta_retriever.get_completion("What changed in the project plan?")
    print(f"Temporal Delta:\n{delta}\n")

if __name__ == "__main__":
    setup_logging(log_level=INFO)
    asyncio.run(main())
