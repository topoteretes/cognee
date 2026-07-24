from enum import Enum
from typing import Any, Dict, List, Tuple

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.models.FieldResolution import FieldResolution


class MergeStrategy(Enum):
    LONGEST = "longest"
    SHORTEST = "shortest"
    LATEST = "latest"


def default_rules() -> Dict[str, MergeStrategy]:
    return {
        "description": MergeStrategy.LONGEST,
        "name": MergeStrategy.LONGEST, # If aliasing causes same ID, keep longest name
    }


class MergePolicy:
    def __init__(self, rules: Dict[str, MergeStrategy] = None):
        self._rules = rules or default_rules()

    def merge_nodes(self, survivor: DataPoint, absorbed: DataPoint) -> Tuple[DataPoint, List[FieldResolution]]:
        """Merge 'absorbed' into 'survivor' based on rules. Both are live DataPoint objects."""
        resolutions = []
        for field_name, strategy in self._rules.items():
            if not hasattr(survivor, field_name) or not hasattr(absorbed, field_name):
                continue
                
            survivor_val = getattr(survivor, field_name)
            absorbed_val = getattr(absorbed, field_name)
            
            if absorbed_val is None:
                continue
            if survivor_val is None:
                object.__setattr__(survivor, field_name, absorbed_val)
                continue
                
            winner, used_strategy = self._resolve(strategy, absorbed_val, survivor_val)
            if winner != survivor_val:
                object.__setattr__(survivor, field_name, winner)
                resolutions.append(FieldResolution(
                    field_name=field_name,
                    old_value=str(survivor_val),
                    new_value=str(winner),
                    strategy=used_strategy,
                ))
        return survivor, resolutions

    def merge_with_existing(
        self, incoming_node: DataPoint, existing_props: dict
    ) -> List[FieldResolution]:
        """Merge incoming node properties with stored properties from get_nodes().

        existing_props is a flat dict from _parse_node_row:
          {"id": ..., "name": ..., "type": ..., <all JSONB fields flattened>}

        Never called for cross-type collisions — id-level separation prevents that.
        Mutates incoming_node in-place.
        """
        field_resolutions = []
        for field_name, strategy in self._rules.items():
            if not hasattr(incoming_node, field_name):
                continue
            incoming_val = getattr(incoming_node, field_name)
            existing_val = existing_props.get(field_name)   # flat, no fallback needed
            if existing_val is None:
                continue
                
            # Treat existing as "survivor" and incoming as "absorbed" for resolution
            winner, used_strategy = self._resolve(strategy, incoming_val, existing_val)
            if winner != incoming_val:
                object.__setattr__(incoming_node, field_name, winner)
                field_resolutions.append(FieldResolution(
                    field_name=field_name,
                    old_value=str(incoming_val),
                    new_value=str(winner),
                    strategy=used_strategy,
                ))
        return field_resolutions

    def _resolve(self, strategy: MergeStrategy, val1: Any, val2: Any) -> Tuple[Any, str]:
        """Resolve a conflict between two values using the given strategy."""
        if strategy == MergeStrategy.LONGEST:
            str1, str2 = str(val1), str(val2)
            if len(str1) > len(str2):
                return val1, strategy.value
            return val2, strategy.value
        elif strategy == MergeStrategy.SHORTEST:
            str1, str2 = str(val1), str(val2)
            if len(str1) < len(str2):
                return val1, strategy.value
            return val2, strategy.value
        # Default fallback to just picking the second value (LATEST)
        return val2, strategy.value
