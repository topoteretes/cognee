"""Shared structural-tag vocabulary.

Every agent is instructed to tag findings using ONLY these descriptors, so
tag-vocabulary drift cannot silently defeat structural retrieval across
domains with non-overlapping natural-language surface.
"""

STRUCTURAL_TAGS: tuple[str, ...] = (
    "independent_steps",
    "memoryless",
    "continuous_time",
    "discrete_time",
    "gaussian_increments",
    "zero_mean",
    "variance_linear_in_t",
    "drift_term",
    "martingale",
    "markov_property",
    "scale_invariant",
    "bounded_above",
    "exponential_growth",
    "log_normal",
    "stationary_increments",
)


def format_for_prompt() -> str:
    """Render the vocabulary as a comma-separated list for system prompts."""
    return ", ".join(STRUCTURAL_TAGS)
