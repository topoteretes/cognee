SOURCE_REF_PREFIX = "source_ref:v1"
SOURCE_RUN_REF_PREFIX = "source_run_ref:v1"

GRAPH_PROVENANCE_VERSION_KEY = "provenance_version"
GRAPH_DELETE_MODE_KEY = "delete_mode"
GRAPH_PROVENANCE_VERSION = "1"
# Stored marker value written into a graph's metadata. The string stays
# "graph_native" as a stable on-disk token (changing it would orphan graphs
# already marked in the experimental release); the code term is "graph provenance".
GRAPH_DELETE_MODE_GRAPH_PROVENANCE = "graph_native"
