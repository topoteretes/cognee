from .fingerprint import StructuralFingerprint, structural_similarity, build_fingerprint
from .candidate_generation import generate_candidate_pairs
from .contradiction_detection import detect_contradictions, Contradiction
from .merge_execution import resolve_structural_duplicates, apply_structural_merges, MergeCandidate
from .undo import undo_merge

__all__ = [
    "StructuralFingerprint",
    "structural_similarity",
    "build_fingerprint",
    "generate_candidate_pairs",
    "detect_contradictions",
    "Contradiction",
    "resolve_structural_duplicates",
    "apply_structural_merges",
    "MergeCandidate",
    "undo_merge",
]