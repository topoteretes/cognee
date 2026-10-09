"""Shared constants for the data/dataset layer."""

# The dataset used whenever a caller states no dataset at all. Cognee's
# convention across add/cognify/remember/improve/memify is "no dataset
# stated -> main_dataset"; session scoping follows the same rule so an
# omitted dataset still produces a dataset-scoped session rather than an
# unscoped, unenforceable one.
DEFAULT_DATASET_NAME = "main_dataset"
