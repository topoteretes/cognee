import asyncio
import json
import time
from dataclasses import dataclass
from aiokafka import AIOKafkaProducer

from .corpus import CorpusGenerator

@dataclass
class ProduceResult:
    messages_sent: int
    bytes_sent: int
    duration_seconds: float
    actual_rate: float

async def produce_corpus(
    bootstrap_servers: str,
    topic: str,
    corpus: CorpusGenerator,
    messages_per_second: float = 0,
    dry_run: bool = False,
) -> ProduceResult:
    """
    Produce the given corpus to a Kafka topic at a specific rate.
    
    If dry_run is True, messages are generated and serialized, but not sent to Kafka.
    """
    producer = None
    if not dry_run:
        producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
        await producer.start()

    messages_sent = 0
    bytes_sent = 0
    start_time = time.perf_counter()

    interval = 1.0 / messages_per_second if messages_per_second > 0 else 0
    next_time = time.perf_counter() + interval if interval > 0 else 0

    try:
        futures = []
        for doc in corpus.generate_stream():
            # Include produced_at in metadata (in microseconds) for end-to-end latency calculation
            produced_at_us = int(time.time() * 1_000_000)
            doc.metadata["produced_at"] = produced_at_us
            
            payload = {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
            
            message_bytes = json.dumps(payload).encode('utf-8')
            
            if not dry_run and producer is not None:
                # send() buffers the message and returns a Future for delivery confirmation
                fut = await producer.send(topic, message_bytes)
                futures.append(fut)
                
                # Await in batches to avoid unbounded memory growth on huge corpora
                if len(futures) >= 1000:
                    await asyncio.gather(*futures)
                    futures.clear()
            
            messages_sent += 1
            bytes_sent += len(message_bytes)
            
            # Rate limiting logic
            if interval > 0:
                current_time = time.perf_counter()
                if current_time < next_time:
                    await asyncio.sleep(next_time - current_time)
                # Recalculate next_time based on actual current time to prevent drift buildup
                next_time = time.perf_counter() + interval
                
        # Wait for any remaining delivery confirmations
        if futures and not dry_run:
            await asyncio.gather(*futures)

    finally:
        if producer is not None:
            await producer.stop()
            
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    return ProduceResult(
        messages_sent=messages_sent,
        bytes_sent=bytes_sent,
        duration_seconds=duration,
        actual_rate=messages_sent / duration if duration > 0 else 0.0
    )
