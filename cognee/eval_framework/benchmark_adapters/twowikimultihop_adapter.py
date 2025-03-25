from typing import Any
from cognee.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter


class TwoWikiMultihopAdapter(HotpotQAAdapter):
    dataset_info = {
        "filename": "2wikimultihop_dev.json",
        "url": "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json",
    }

    def __init__(self):
        super().__init__()
        self.metadata_field_name = "type"

    def _get_golden_context(self, item: dict[str, Any]) -> str:
        """Extracts and formats the golden context from supporting facts and adds evidence if available."""
        golden_context = super()._get_golden_context(item)

        if "evidences" in item:
            golden_context += "\nEvidence fact triplets:"
            for subject, relation, obj in item["evidences"]:
                golden_context += f"\n  • {subject} - {relation} - {obj}"

        return golden_context
