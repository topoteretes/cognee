from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union, Tuple
import os
import json


class BaseBenchmarkAdapter(ABC):
    def _filter_instances(
        self,
        instances: List[dict[str, Any]],
        instance_filter: Union[str, List[str], List[int]],
        id_key: str = "id",
    ) -> List[dict[str, Any]]:
        """
        Filter instances by IDs or indices, or load filter from a JSON file.

        Args:
            instances: List of instances to filter
            instance_filter: Filter criteria (IDs, indices, or path to JSON file)
            id_key: The key used for ID in the instances (defaults to "id")

        Returns:
            Filtered list of instances

        Raises:
            FileNotFoundError: If filter file not found
            ValueError: If filter format is invalid
        """
        if isinstance(instance_filter, str):
            if not os.path.isfile(instance_filter):
                raise FileNotFoundError(f"Filter file not found: {instance_filter}")

            with open(instance_filter, "r", encoding="utf-8") as f:
                try:
                    instance_filter = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in filter file: {e}")

        if all(isinstance(fid, str) for fid in instance_filter):
            return [inst for inst in instances if inst.get(id_key) in instance_filter]

        if all(isinstance(fid, int) for fid in instance_filter):
            return [instances[i] for i in instance_filter if 0 <= i < len(instances)]

        raise ValueError(
            "instance_filter must be a list of string ids, integer indices, or a JSON file path."
        )

    @abstractmethod
    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        pass
