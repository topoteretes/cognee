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
# Same reasoning as rewrite_entities.PARAGRAPH_MAX_COMPLETION_TOKENS - both
# query_type_LLM and query_type_merge_LLM produce a single short paragraph.
PARAGRAPH_MAX_COMPLETION_TOKENS = 250
# query_is_a_only_LLM produces one short is_a line per member in the batch,
# not a single paragraph - the budget scales with how many members are in
# that specific call rather than being a single fixed constant.
TOKENS_PER_IS_A_LINE = 60
