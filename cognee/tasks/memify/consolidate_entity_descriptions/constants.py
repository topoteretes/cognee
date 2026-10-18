def truncate(text: str, max_chars: int) -> str:
    """Cap a string for a prompt line or a persisted edge property."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


prompt_name = "consolidate_entity_details.txt"
type_prompt_name = "consolidate_entity_type_details.txt"
type_merge_prompt_name = "consolidate_entity_type_merge.txt"
is_a_only_prompt_name = "consolidate_entity_is_a_only.txt"
MAX_CONCURRENT_ENTITY_LLM_CALLS = 10
MAX_CONCURRENT_TYPE_LLM_CALLS = 10
# Counted in prompt LINES, not neighbors: one neighbor contributes one line per
# distinct edge connecting it, so a neighbor cap bounds nothing an
# over-connected entity can do to the prompt.
MAX_NEIGHBOR_LINES_IN_PROMPT = 20
MAX_NEIGHBOR_TEXT_CHARS = 500
MAX_NAMED_MEMBERS = 5
MAX_MEMBERS_PER_TYPE_PROMPT = 50
# One cap per job, because the three jobs bound different things.
#
# A member card is an arbitrary phase-1 description shown as one prompt line -
# same defensive cap as rewrite_entities.MAX_NEIGHBOR_TEXT_CHARS.
MAX_MEMBER_CARD_CHARS = 500
# What actually lands on the is_a edge. Independent of the LLM call's output
# budget: that budget caps generation, this caps what reaches the graph.
MAX_PERSISTED_IS_A_CHARS = 500
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
# Every paragraph-producing call in this pipeline - the entity rewrite, the
# type summary and the merge - returns one short paragraph (~500 chars, ~125
# tokens). One budget for all three: separate copies drift, and a change meant
# for one stage silently leaves the others behind.
PARAGRAPH_MAX_COMPLETION_TOKENS = REASONING_HEADROOM_TOKENS + 250
# A merge partial is this pipeline's OWN phase-2 output, not untrusted text:
# it was generated under the content half of the budget above. Cutting it to
# the member-card cap would throw away half of every partial before the merge
# call reads it, so derive the cap from the budget that produced it (~4 chars
# per token) instead of picking a second number that can drift from it.
MAX_MERGE_PARTIAL_CHARS = (PARAGRAPH_MAX_COMPLETION_TOKENS - REASONING_HEADROOM_TOKENS) * 4
# query_is_a_only_LLM produces one short is_a line per member in the batch,
# not a single paragraph - the content portion of the budget scales with how
# many members are in that specific call; REASONING_HEADROOM_TOKENS is added
# separately at the call site so small batches still get the same floor.
TOKENS_PER_IS_A_LINE = 60
