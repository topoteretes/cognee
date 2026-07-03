"""
Kafka consumer bridge for cognee.
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass
from typing import Optional, List
from aiokafka import AIOKafkaConsumer

import cognee

logger = logging.getLogger(__name__)

@dataclass
class BatchResult:
    """
    Result of processing a single batch of messages from Kafka.
    
    States what it does: Holds nanosecond-precision metrics for a single ingestion batch.
    States what cognee API it calls: It does not call any cognee API itself.
    States what it explicitly does NOT do: It does not modify the cognee pipeline runner; it is a simple data container.
    """
    start_ns: int
    end_ns: int
    message_count: int
    bytes_processed: int
    add_latency_ns: int
    cognify_latency_ns: int
    success: bool
    error: Optional[str] = None


class KafkaCogneeConsumer:
    """
    Async Kafka consumer bridge for Cognee.
    
    States what it does: Consumes messages from a Kafka topic and ingests them into a Cognee dataset.
    States what cognee API it calls: Calls cognee.add() and cognee.cognify().
    States what it explicitly does NOT do: Does not modify the cognee pipeline runner; calls add() and cognify() through the public API.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        dataset_name: str,
        batch_size: int = 10,
        max_concurrency: int = 4,
        backpressure_threshold: int = 100
    ):
        """
        Initializes the consumer bridge.
        
        States what it does: Sets up configuration, the concurrency semaphore, and the results queue.
        States what cognee API it calls: Does not call any cognee API during initialization.
        States what it explicitly does NOT do: Does not modify the cognee pipeline runner; calls add() and cognify() through the public API.
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.backpressure_threshold = backpressure_threshold
        
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self.results_queue: asyncio.Queue[BatchResult] = asyncio.Queue(maxsize=self.backpressure_threshold)
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._tasks: set[asyncio.Task] = set()

    async def _process_batch(self, batch_data: List[str], batch_bytes: int) -> BatchResult:
        """
        Processes a batch of messages.
        
        States what it does: Executes the ingestion process for a given batch of documents and records nanosecond-precision timings for total, add, and cognify latencies.
        States what cognee API it calls: Calls cognee.add() with the batch content and cognee.cognify() for the dataset.
        States what it explicitly does NOT do: Does not modify the cognee pipeline runner; calls add() and cognify() through the public API.
        """
        start_ns = time.perf_counter_ns()
        add_start_ns = 0
        add_end_ns = 0
        cognify_end_ns = 0
        success = True
        error_msg = None

        try:
            add_start_ns = time.perf_counter_ns()
            await cognee.add(batch_data, dataset_name=self.dataset_name)
            add_end_ns = time.perf_counter_ns()
            
            await cognee.cognify(datasets=[self.dataset_name])
            cognify_end_ns = time.perf_counter_ns()
        except Exception as e:
            logger.exception("Failed to process batch in cognee")
            success = False
            error_msg = str(e)
            if add_end_ns == 0:
                add_end_ns = time.perf_counter_ns()
            if cognify_end_ns == 0:
                cognify_end_ns = time.perf_counter_ns()

        end_ns = time.perf_counter_ns()

        return BatchResult(
            start_ns=start_ns,
            end_ns=end_ns,
            message_count=len(batch_data),
            bytes_processed=batch_bytes,
            add_latency_ns=add_end_ns - add_start_ns if add_start_ns > 0 else 0,
            cognify_latency_ns=cognify_end_ns - add_end_ns if add_end_ns > 0 else 0,
            success=success,
            error=error_msg
        )

    async def _dispatch_batch(self, batch_data: List[str], batch_bytes: int):
        """
        Dispatches the batch for processing and releases the semaphore when done.
        
        States what it does: Runs _process_batch, pushes the BatchResult to the results_queue, and releases the semaphore to allow the next batch.
        States what cognee API it calls: Indirectly calls cognee.add() and cognee.cognify() via _process_batch().
        States what it explicitly does NOT do: Does not modify the cognee pipeline runner; calls add() and cognify() through the public API.
        """
        try:
            result = await self._process_batch(batch_data, batch_bytes)
            await self.results_queue.put(result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected error in _dispatch_batch")
        finally:
            self._semaphore.release()

    async def run(self):
        """
        Main consumer loop.
        
        States what it does: Accumulates messages into batches of batch_size from AIOKafkaConsumer. When a batch is full (or a timeout fires), it acquires the concurrency semaphore and spawns _dispatch_batch. Handles cancellation cleanly.
        States what cognee API it calls: Indirectly calls cognee.add() and cognee.cognify() for the dataset.
        States what it explicitly does NOT do: Does not modify the cognee pipeline runner; calls add() and cognify() through the public API.
        """
        self._consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="earliest"
        )
        
        await self._consumer.start()
        logger.info("Started KafkaCogneeConsumer on topic %s", self.topic)
        
        batch_data = []
        batch_bytes = 0
        
        try:
            while True:
                records = await self._consumer.getmany(timeout_ms=1000, max_records=self.batch_size - len(batch_data))
                
                for tp, messages in records.items():
                    for msg in messages:
                        text_content = msg.value.decode('utf-8', errors='replace')
                        try:
                            payload = json.loads(text_content)
                            if isinstance(payload, dict) and "text" in payload:
                                text_content = payload["text"]
                        except json.JSONDecodeError:
                            pass
                            
                        batch_data.append(text_content)
                        batch_bytes += len(msg.value)
                
                if len(batch_data) >= self.batch_size or (not records and len(batch_data) > 0):
                    await self._semaphore.acquire()
                    
                    task = asyncio.create_task(self._dispatch_batch(batch_data, batch_bytes))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                    
                    batch_data = []
                    batch_bytes = 0
                    
        except asyncio.CancelledError:
            logger.info("KafkaCogneeConsumer cancelled, initiating shutdown")
        except Exception:
            logger.exception("Fatal error in KafkaCogneeConsumer loop")
            raise
        finally:
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            if self._consumer:
                await self._consumer.stop()
            logger.info("KafkaCogneeConsumer shutdown complete")
