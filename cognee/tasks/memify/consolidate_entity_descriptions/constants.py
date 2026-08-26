type_prompt_name = "consolidate_entity_type_details.txt"
type_merge_prompt_name = "consolidate_entity_type_merge.txt"
is_a_only_prompt_name = "consolidate_entity_is_a_only.txt"
MAX_CONCURRENT_TYPE_LLM_CALLS = 10
MAX_NAMED_MEMBERS = 5
MAX_MEMBERS_PER_TYPE_PROMPT = 50
# Same cap as rewrite_entities.MAX_NEIGHBOR_TEXT_CHARS - a member card or a
# merge partial is the same kind of content (a compact description paragraph)
# shown the same way (one item per line), just in a different phase of the
# pipeline.
MAX_TYPE_TEXT_CHARS = 500
# Every completion-token budget in this pipeline needs headroom beyond its
# visible-content target: reasoning models (cognee's own default, gpt-5-mini,
# among them) spend hidden reasoning tokens out of the same
# max_completion_tokens budget, and a text-length estimate can't see or bound
# those. Confirmed empirically against the live default model: a 250-token
# budget with no headroom intermittently returned empty content (reasoning
# alone exhausted it), while adding this much headroom succeeded with zero
# retries. Added on top of every budget below, including the per-member is_a
# budget - a small batch needs the same floor a large one does, since
# reasoning cost doesn't scale down with fewer visible output lines.
REASONING_HEADROOM_TOKENS = 2000
# Same reasoning as rewrite_entities.PARAGRAPH_MAX_COMPLETION_TOKENS - both
# query_type_LLM and query_type_merge_LLM produce a single short paragraph
# (~500 chars, ~125 tokens); the rest of the budget is reasoning headroom.
PARAGRAPH_MAX_COMPLETION_TOKENS = REASONING_HEADROOM_TOKENS + 250
# query_is_a_only_LLM produces one short is_a line per member in the batch,
# not a single paragraph - the content portion of the budget scales with how
# many members are in that specific call; REASONING_HEADROOM_TOKENS is added
# separately at the call site so small batches still get the same floor.
TOKENS_PER_IS_A_LINE = 60
