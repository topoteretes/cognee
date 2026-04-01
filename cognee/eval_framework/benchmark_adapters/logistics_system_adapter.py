from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from cognee.eval_framework.benchmark_adapters.base_benchmark_adapter import (
    BaseBenchmarkAdapter,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.corpus_generator.narrativize_corpus import (
    PACKAGE_FOLDER,
    load_narrative_corpus,
    narrativize_corpus,
    narrative_corpus_exists,
    run_narrativize_corpus,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.ontology import (
    add_packages_to_world,
    create_world,
    validate_world_packages,
    write_golden_answers,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.utils.utils import (
    _safe_filename,
    load_world,
    store_world,
)


DEFAULT_WORLD_ROOT = Path("data/logistics_system_worlds")
DEFAULT_WORLD_NAME = "default"
DEFAULT_WORLD_FILENAME = "stored_world.json"
DEFAULT_GOLDEN_ANSWERS_FILENAME = "golden_answers.json"
DEFAULT_QUERY_PATH = (
    Path(__file__).resolve().parent / "logistics_system_utils" / "queries" / "query.txt"
)
DEFAULT_COMMON_RULES_PATH = (
    Path(__file__).resolve().parent
    / "logistics_system_utils"
    / "rules"
    / "rules_common_language.txt"
)


class LogisticsSystemAdapter(BaseBenchmarkAdapter):
    def __init__(
        self,
        world_name: str = DEFAULT_WORLD_NAME,
        worlds_root: str | Path = DEFAULT_WORLD_ROOT,
        user_count: int = 15,
        retailer_count: int = 10,
        package_count: int = 5,
        query: str | None = None,
    ) -> None:
        super().__init__()
        self.world_name = world_name
        self.worlds_root = Path(worlds_root)
        self.user_count = user_count
        self.retailer_count = retailer_count
        self.package_count = package_count
        self.query = query if query is not None else DEFAULT_QUERY_PATH.read_text(encoding="utf-8")

    @property
    def world_directory(self) -> Path:
        return self.worlds_root / self.world_name

    @property
    def world_file(self) -> Path:
        return self.world_directory / DEFAULT_WORLD_FILENAME

    @property
    def golden_answers_file(self) -> Path:
        return self.world_directory / DEFAULT_GOLDEN_ANSWERS_FILENAME

    def _create_world_with_packages(self, max_attempts: int = 10) -> dict[str, object]:
        for _ in range(max_attempts):
            world = create_world(
                user_count=self.user_count,
                retailer_count=self.retailer_count,
            )
            try:
                return add_packages_to_world(world, package_count=self.package_count)
            except ValueError:
                continue

        raise ValueError(
            "Could not generate a logistics world with compatible user and retailer pairs."
        )

    def _get_or_create_world(self) -> dict[str, object]:
        if self.world_file.exists():
            world = load_world(self.world_file)
            validate_world_packages(world)
            if not self.golden_answers_file.exists():
                write_golden_answers(world, self.world_directory)
            return world

        self.world_directory.mkdir(parents=True, exist_ok=True)
        world = self._create_world_with_packages()
        store_world(world, self.world_file)
        write_golden_answers(world, self.world_directory)
        return world

    async def prepare_corpus(self) -> None:
        world = self._get_or_create_world()
        if not narrative_corpus_exists(self.world_directory, world):
            await narrativize_corpus(self.world_directory)

    def _load_golden_answers(self) -> list[dict[str, Any]]:
        payload = json.loads(self.golden_answers_file.read_text(encoding="utf-8"))
        return payload.get("golden_answers", [])

    def _load_package_narratives(self, world: dict[str, object]) -> dict[str, str]:
        if not narrative_corpus_exists(self.world_directory, world):
            run_narrativize_corpus(self.world_directory)

        package_narratives: dict[str, str] = {}
        for package in world.get("packages", []):
            narrative_path = (
                self.world_directory / PACKAGE_FOLDER / f"{_safe_filename(package.package_id)}.txt"
            )
            if not narrative_path.exists():
                raise ValueError(
                    f"Missing package narrative for {package.package_id} at {narrative_path}."
                )
            package_narratives[package.package_id] = narrative_path.read_text(
                encoding="utf-8"
            ).strip()

        return package_narratives

    def _load_shared_corpus_documents(self) -> list[str]:
        return [DEFAULT_COMMON_RULES_PATH.read_text(encoding="utf-8").strip()]

    def _build_question_answer_pairs(
        self,
        world: dict[str, object],
        load_golden_context: bool = False,
    ) -> List[dict[str, Any]]:
        packages = list(world.get("packages", []))
        golden_answers = self._load_golden_answers()
        package_narratives = self._load_package_narratives(world)

        if len(packages) != len(golden_answers):
            raise ValueError(
                "The number of generated packages does not match the number of golden answers."
            )

        qa_pairs: list[dict[str, Any]] = []
        question_specs = (
            (
                "delivery_days",
                "Estimate the total delivery days for this package.",
                "estimated_delivery_days",
                "estimated_delivery_days_supporting_facts",
                "estimated_delivery_days_supporting_facts_data_sources",
            ),
            (
                "transport_cost",
                "Estimate the transportation cost for this package.",
                "estimated_transport_price",
                "estimated_transport_price_supporting_facts",
                "estimated_transport_price_supporting_facts_data_sources",
            ),
            (
                "carrier",
                "Estimate the most suitable carrier for this package.",
                "selected_carrier",
                "carrier_selection_reasons",
                "carrier_selection_reasons_data_sources",
            ),
        )

        for package, golden_answer in zip(packages, golden_answers, strict=False):
            package_context = package_narratives[package.package_id]
            for (
                question_type,
                question_prompt,
                answer_key,
                supporting_facts_key,
                data_sources_key,
            ) in question_specs:
                question_text = f"{package_context}\n\n{question_prompt}"
                # if self.query.strip():
                #    question_text = f"{question_text}\n{self.query.strip()}"

                qa_pair = {
                    "id": f"{self.world_name}:{package.package_id}:{question_type}",
                    "question": question_text,
                    "answer": str(golden_answer.get(answer_key)),
                    "golden_answer": golden_answer.get(answer_key),
                    "golden_context": "\n".join(golden_answer.get(supporting_facts_key, [])),
                    "golden_context_data_sources": golden_answer.get(data_sources_key, []),
                    "type": "logistics_system",
                    "question_type": question_type,
                    "world_name": self.world_name,
                    "package_id": package.package_id,
                }
                if load_golden_context:
                    qa_pair["package_context"] = package_context
                qa_pairs.append(qa_pair)

        return qa_pairs

    def load_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = False,
        instance_filter: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        world = self._get_or_create_world()
        if not narrative_corpus_exists(self.world_directory, world):
            run_narrativize_corpus(self.world_directory)
        corpus_list = self._load_shared_corpus_documents() + load_narrative_corpus(
            self.world_directory
        )
        qa_pairs = self._build_question_answer_pairs(world, load_golden_context)

        if instance_filter is not None:
            qa_pairs = self._filter_instances(qa_pairs, instance_filter, id_key="id")

        if limit is not None and 0 < limit < len(qa_pairs):
            random.seed(seed)
            qa_pairs = random.sample(qa_pairs, limit)

        return corpus_list, qa_pairs
