from enum import Enum
from typing import Type

from cognee.eval_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
from cognee.eval_framework.benchmark_adapters.logistics_system_adapter import (
    LogisticsSystemAdapter,
)
from cognee.eval_framework.benchmark_adapters.musique_adapter import MusiqueQAAdapter
from cognee.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter
from cognee.eval_framework.benchmark_adapters.twowikimultihop_adapter import TwoWikiMultihopAdapter
from cognee.eval_framework.benchmark_adapters.beam_adapter import BEAMAdapter


class BenchmarkAdapter(Enum):
    DUMMY = ("Dummy", DummyAdapter)
    HOTPOTQA = ("HotPotQA", HotpotQAAdapter)
    LOGISTICSSYSTEM = ("LogisticsSystem", LogisticsSystemAdapter)
    MUSIQUE = ("Musique", MusiqueQAAdapter)
    TWOWIKIMULTIHOP = ("TwoWikiMultiHop", TwoWikiMultihopAdapter)
    BEAM = ("BEAM", BEAMAdapter)

    def __new__(cls, adapter_name: str, adapter_class: Type):
        obj = object.__new__(cls)
        obj._value_ = adapter_name
        obj.adapter_class = adapter_class
        return obj

    def __str__(self):
        return self.value
