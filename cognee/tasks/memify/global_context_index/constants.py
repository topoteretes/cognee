SUMMARIZED_IN = "summarized_in"
GLOBAL_CONTEXT_SUMMARY_COLLECTION = "GlobalContextSummary_text"
TEXT_SUMMARY_COLLECTION = "TextSummary_text"
SUMMARY_GRAPH_NODE_TYPES = ["TextSummary", "DocumentChunk", "GlobalContextSummary"]

# Deterministic document index (method="from_entities").
SUMMARIZES = "summarizes"
DOCUMENT_INDEX_NODE_TYPES = ["DocumentChunk", "Entity", "EntityType", "GlobalContextSummary"]
DEFAULT_ENTITY_TYPE_NAME = "Entity"
# The longest entity or type name that bucket and root text state; a longer one is cut.
ENTITY_NAME_MAX_CHARS = 80
# Small on purpose: bucket and root texts stay embeddable and cheap to paste into a prompt.
# Relations get the larger share of that budget: a relation line states two names and what
# holds between them, so it says more per character than another bare name would.
DOCUMENT_BUCKET_MAX_ENTITIES = 20
DOCUMENT_BUCKET_MAX_RELATIONS = 40
ROOT_MAX_ENTITIES = 40
ROOT_MAX_TYPES = 20
ROOT_MAX_DOCUMENTS = 40
