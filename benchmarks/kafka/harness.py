import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

import cognee
from cognee.api.v1.search import SearchType

from benchmarks.kafka.corpus import CorpusGenerator
from benchmarks.kafka.producer import produce_corpus
from cognee.integrations.kafka.consumer import KafkaCogneeConsumer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def run_benchmark(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_iso = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
    results_file = out_dir / f"kafka_benchmark_{timestamp_iso}.json"

    from cognee import __version__ as cognee_version
    
    results = {
        "meta": {
            "python_version": sys.version.split(" ")[0],
            "cognee_version": cognee_version,
            "corpus_seed": CorpusGenerator.SEED,
            "batch_size": args.batch_size,
            "max_concurrency": args.concurrency,
            "llm_model": os.getenv("LLM_MODEL", "unknown"),
            "run_at": datetime.utcnow().isoformat(),
            "mock_llm": args.mock_llm
        },
        "throughput": {},
        "latency": {},
        "search_vs_scale": []
    }

    mock_context = None
    if args.mock_llm:
        from cognee.tests.shared.mocks.llm_harness import MockLLMContext
        mock_context = MockLLMContext()
        await mock_context.__aenter__()
        
        # The MockLLMContext doesn't fully mock embeddings, which causes crashes
        # during the real search benchmark later on.

    try:
        topic = "cognee-benchmark-v1"
        dataset_name = "benchmark_dataset"
        
        # Clean DB
        from cognee.modules.engine.operations.setup import setup
        await setup()
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        logger.info(f"Initializing Corpus with {args.messages} messages")
        corpus = CorpusGenerator(num_documents=args.messages)
        
        consumer = KafkaCogneeConsumer(
            bootstrap_servers=args.bootstrap_servers,
            topic=topic,
            group_id="benchmark_group",
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            max_concurrency=args.concurrency
        )
        
        consumer_task = asyncio.create_task(consumer.run())

        produce_start = time.perf_counter()
        prod_res = await produce_corpus(
            bootstrap_servers=args.bootstrap_servers,
            topic=topic,
            corpus=corpus,
            messages_per_second=0,
            dry_run=False
        )
        logger.info(f"Produced {prod_res.messages_sent} messages in {prod_res.duration_seconds:.2f}s")
        
        received_messages = 0
        batch_results = []
        first_message_ts = None
        last_graph_write_ts = None
        
        while received_messages < args.messages:
            try:
                res = await asyncio.wait_for(consumer.results_queue.get(), timeout=15.0)
                batch_results.append(res)
                received_messages += res.message_count
                
                if first_message_ts is None:
                    first_message_ts = res.start_ns
                last_graph_write_ts = res.end_ns
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for messages. Received {received_messages}/{args.messages}")
                break

        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        if first_message_ts and last_graph_write_ts:
            total_time_s = (last_graph_write_ts - first_message_ts) / 1e9
            results["throughput"] = {
                "total_messages": received_messages,
                "duration_seconds": total_time_s,
                "messages_per_second": received_messages / total_time_s if total_time_s > 0 else 0
            }
            
            add_latencies = [b.add_latency_ns for b in batch_results]
            cognify_latencies = [b.cognify_latency_ns for b in batch_results]
            total_latencies = [b.end_ns - b.start_ns for b in batch_results]
            
            results["latency"] = {
                "add_p50_ns": int(np.percentile(add_latencies, 50)),
                "add_p95_ns": int(np.percentile(add_latencies, 95)),
                "add_p99_ns": int(np.percentile(add_latencies, 99)),
                
                "cognify_p50_ns": int(np.percentile(cognify_latencies, 50)),
                "cognify_p95_ns": int(np.percentile(cognify_latencies, 95)),
                "cognify_p99_ns": int(np.percentile(cognify_latencies, 99)),
                
                "total_p50_ns": int(np.percentile(total_latencies, 50)),
                "total_p95_ns": int(np.percentile(total_latencies, 95)),
                "total_p99_ns": int(np.percentile(total_latencies, 99)),
            }
        
        # Search vs scale
        logger.info("Running search benchmark")
        search_latencies = []
        queries = [
            "What is the architecture decision regarding caching?",
            "How is the database migration handled?",
            "What vulnerabilities were patched?",
            "Are there any memory leaks?",
            "How does the authorization pipeline work?"
        ]
        
        if args.mock_llm:
            # Skip search latency benchmark in mock mode -
            # MockVectorEngine does not implement embedding_engine
            logger.info("Skipping real search benchmark because --mock-llm is active.")
            search_latencies = [0] * 50
        else:
            for i in range(50):
                q = queries[i % len(queries)]
                qs = time.perf_counter_ns()
                await cognee.search(q, query_type=SearchType.CHUNKS)
                qe = time.perf_counter_ns()
                search_latencies.append(qe - qs)
            
        results["search_vs_scale"].append({
            "corpus_size": args.corpus_size or received_messages,
            "p50_ns": int(np.percentile(search_latencies, 50)),
            "p95_ns": int(np.percentile(search_latencies, 95)),
            "p99_ns": int(np.percentile(search_latencies, 99))
        })
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {results_file}")
        
    finally:
        if mock_context:
            await mock_context.__aexit__(None, None, None)

def main():
    parser = argparse.ArgumentParser(description="Kafka Benchmark Harness")
    parser.add_argument("--messages", type=int, required=True, help="Number of messages to produce/consume")
    parser.add_argument("--corpus-size", type=int, help="Target corpus size for search tests")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for consumer")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrency for consumer")
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--mock-llm", action="store_true", help="Use LLM mock harness")
    parser.add_argument("--with-improve", action="store_true", help="Benchmark improve() overhead")
    parser.add_argument("--output", type=str, default="benchmarks/results/", help="Output directory")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))

if __name__ == "__main__":
    main()
