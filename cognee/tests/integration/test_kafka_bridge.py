import pytest
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock

import cognee
from benchmarks.kafka.corpus import CorpusGenerator
from benchmarks.kafka.producer import produce_corpus
from cognee.integrations.kafka.consumer import KafkaCogneeConsumer
from benchmarks.kafka.harness import run_benchmark

from cognee.tests.shared.mocks.llm_harness import mock_llm_harness

def test_corpus_determinism():
    gen1 = CorpusGenerator(num_documents=100)
    gen2 = CorpusGenerator(num_documents=100)
    
    docs1 = gen1.generate()
    docs2 = gen2.generate()
    
    assert len(docs1) == 100
    assert len(docs2) == 100
    
    # Assert identical output (seed isolation)
    assert docs1[0].id == "doc_000000"
    assert docs1[0].content == docs2[0].content
    assert docs1[50].content == docs2[50].content
    assert docs1[99].metadata == docs2[99].metadata


@pytest.mark.asyncio
async def test_producer_dry_run():
    gen = CorpusGenerator(num_documents=50)
    result = await produce_corpus(
        bootstrap_servers="invalid:9092",
        topic="test",
        corpus=gen,
        dry_run=True
    )
    
    assert result.messages_sent == 50
    assert result.bytes_sent > 0


@pytest.mark.asyncio
async def test_batch_result_shape():
    consumer = KafkaCogneeConsumer(
        bootstrap_servers="localhost",
        topic="test",
        group_id="test",
        dataset_name="test_dataset"
    )
    
    with patch("cognee.add", new_callable=AsyncMock) as mock_add, \
         patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify:
        
        batch_data = ["msg1", "msg2", "msg3", "msg4", "msg5"]
        batch_bytes = 100
        
        result = await consumer._process_batch(batch_data, batch_bytes)
        
        mock_add.assert_called_once()
        mock_cognify.assert_called_once()
        
        assert hasattr(result, "start_ns")
        assert hasattr(result, "end_ns")
        assert hasattr(result, "message_count")
        assert hasattr(result, "bytes_processed")
        assert hasattr(result, "add_latency_ns")
        assert hasattr(result, "cognify_latency_ns")
        assert hasattr(result, "success")
        assert hasattr(result, "error")
        
        assert isinstance(result.add_latency_ns, int)
        assert isinstance(result.cognify_latency_ns, int)
        assert result.add_latency_ns >= 0
        assert result.cognify_latency_ns >= 0
        assert result.message_count == 5
        assert result.bytes_processed == 100
        assert result.success is True


@pytest.mark.asyncio
async def test_backpressure_semaphore():
    consumer = KafkaCogneeConsumer(
        bootstrap_servers="localhost",
        topic="test",
        group_id="test",
        dataset_name="test_dataset",
        max_concurrency=2
    )
    
    assert consumer._semaphore._value == 2
    
    concurrent_processes = 0
    max_concurrent_processes = 0
    lock = asyncio.Lock()
    
    async def mock_process_batch(*args, **kwargs):
        nonlocal concurrent_processes, max_concurrent_processes
        async with lock:
            concurrent_processes += 1
            if concurrent_processes > max_concurrent_processes:
                max_concurrent_processes = concurrent_processes
        
        await asyncio.sleep(0.1)
        
        async with lock:
            concurrent_processes -= 1
            
        from cognee.integrations.kafka.consumer import BatchResult
        return BatchResult(0, 0, 1, 1, 0, 0, True, None)

    with patch.object(consumer, "_process_batch", side_effect=mock_process_batch):
        tasks = []
        for _ in range(10):
            await consumer._semaphore.acquire()
            tasks.append(asyncio.create_task(consumer._dispatch_batch(["msg"], 1)))
            
        await asyncio.gather(*tasks)
        
        assert max_concurrent_processes <= 2


# --- Mock Kafka infrastructure for E2E tests ---
mock_kafka_messages = []

class DummyKafkaProducer:
    def __init__(self, *args, **kwargs):
        pass
    async def start(self):
        pass
    async def stop(self):
        pass
    async def send(self, topic, value):
        mock_kafka_messages.append(value)
        fut = asyncio.Future()
        fut.set_result(None)
        return fut

class DummyKafkaMessage:
    def __init__(self, val):
        self.value = val

class DummyKafkaConsumer:
    def __init__(self, *args, **kwargs):
        pass
    async def start(self):
        pass
    async def stop(self):
        pass
    async def getmany(self, timeout_ms=1000, max_records=10):
        if not mock_kafka_messages:
            await asyncio.sleep(0.1)
            return {}
        batch = mock_kafka_messages[:max_records]
        del mock_kafka_messages[:max_records]
        return {"test_tp": [DummyKafkaMessage(m) for m in batch]}

@pytest.fixture
def mock_kafka():
    mock_kafka_messages.clear()
    with patch("benchmarks.kafka.producer.AIOKafkaProducer", DummyKafkaProducer), \
         patch("cognee.integrations.kafka.consumer.AIOKafkaConsumer", DummyKafkaConsumer):
        yield


@pytest.mark.asyncio
async def test_end_to_end_smoke(mock_llm_harness, mock_kafka):
    import argparse
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    args = argparse.Namespace(
        messages=10,
        corpus_size=10,
        batch_size=5,
        concurrency=2,
        bootstrap_servers="localhost:9092",
        mock_llm=True,
        with_improve=False,
        output=str(out_dir)
    )
    
    # Mock cognee.search so the 50 search queries in the harness don't hit 
    # unmocked embedding endpoints and timeout on the dummy CI API keys.
    with patch("benchmarks.kafka.harness.cognee.search", new_callable=AsyncMock) as mock_search:
        await run_benchmark(args)
        assert mock_search.call_count == 50
    
    json_files = list(out_dir.glob("kafka_benchmark_*.json"))
    assert len(json_files) >= 1
    
    latest = max(json_files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
        
    assert data["throughput"]["messages_per_second"] >= 0
    assert data["latency"]["cognify_p50_ns"] >= 0
    assert len(data["search_vs_scale"]) == 1
    assert data["search_vs_scale"][0]["corpus_size"] == 10


def test_results_json_schema():
    out_dir = Path("benchmarks/results")
    json_files = list(out_dir.glob("kafka_benchmark_*.json"))
    assert len(json_files) >= 1
    
    latest = max(json_files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
        
    assert "meta" in data
    assert "throughput" in data
    assert "latency" in data
    assert "search_vs_scale" in data
    
    assert data["meta"]["mock_llm"] is True
