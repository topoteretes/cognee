import os
from typing import Optional, Type, List
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever

# Get the directory where this file resides
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

ACTION_ITEMS_PROMPT_PATH = os.path.join(CURRENT_DIR, "prompts", "notetaker_action_items.txt")
DECISIONS_PROMPT_PATH = os.path.join(CURRENT_DIR, "prompts", "notetaker_decisions.txt")
TEMPORAL_DELTA_PROMPT_PATH = os.path.join(CURRENT_DIR, "prompts", "notetaker_temporal_delta.txt")


class NotetakerActionItemRetriever(TemporalRetriever):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt_path=ACTION_ITEMS_PROMPT_PATH,
            **kwargs
        )


class NotetakerDecisionRetriever(TemporalRetriever):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt_path=DECISIONS_PROMPT_PATH,
            **kwargs
        )


class NotetakerTemporalDeltaRetriever(TemporalRetriever):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt_path=TEMPORAL_DELTA_PROMPT_PATH,
            **kwargs
        )
