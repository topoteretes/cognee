import asyncio
import multiprocessing

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter


def _hold_lock(url, attempting, entered, release):
    adapter = LanceDBAdapter(url=url, api_key=None, embedding_engine=None)

    async def run():
        attempting.set()
        async with adapter._write_lock():
            entered.set()
            release.wait(timeout=10)

    asyncio.run(run())


def test_local_store_writers_are_serialized_across_processes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    url = str(tmp_path / "lancedb")
    # Every Event must stay referenced here: Process.start() drops its args once
    # they are pickled, and a collected Event unlinks the semaphore the child
    # has not rebuilt yet.
    first_attempting, first_entered, first_release = ctx.Event(), ctx.Event(), ctx.Event()
    second_attempting, second_entered, second_release = ctx.Event(), ctx.Event(), ctx.Event()
    first = ctx.Process(
        target=_hold_lock, args=(url, first_attempting, first_entered, first_release)
    )
    second = ctx.Process(
        target=_hold_lock,
        args=(url, second_attempting, second_entered, second_release),
    )
    try:
        first.start()
        assert first_entered.wait(30)
        second.start()
        assert second_attempting.wait(30)
        assert not second_entered.wait(0.2)
        first_release.set()
        assert second_entered.wait(10)
    finally:
        first_release.set()
        second_release.set()
        for process in (first, second):
            if process.pid:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join()
    assert first.exitcode == second.exitcode == 0


class _HashEmbeddingEngine:
    """Deterministic embeddings, constructible in a spawned child."""

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[len(text) / 100 + 0.01, 0.5, 0.5] for text in texts]


def _note_type():
    # Built inside each process: a DataPoint subclass is not picklable by value.
    from cognee.infrastructure.engine import DataPoint

    class Note(DataPoint):
        text: str
        metadata: dict = {"index_fields": ["text"]}

    return Note


def _upsert_notes(url, ids, start, done):
    Note = _note_type()
    adapter = LanceDBAdapter(url=url, api_key=None, embedding_engine=_HashEmbeddingEngine())

    async def run():
        notes = [Note(id=note_id, text=f"note {note_id}") for note_id in ids]
        start.wait(timeout=60)
        # Small batches, so the processes interleave many commits.
        for offset in range(0, len(notes), 5):
            await adapter.create_data_points("Note_text", notes[offset : offset + 5])
        done.set()

    asyncio.run(run())


def _join(processes):
    for process in processes:
        if process.pid:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join()


def test_concurrent_upserts_from_several_processes_leave_no_duplicate_ids(tmp_path):
    """The workload from #5254: racing processes upserting the same new ids.

    Without a store-wide lock a process's merge_insert can miss rows another
    one is committing and insert its own copy, or fail with "Too many
    concurrent writers". The race does not reproduce on every run, so this is
    a smoke test of the locked path; the lock-wait tests below are the
    deterministic guards.
    """
    from uuid import uuid4

    ctx = multiprocessing.get_context("spawn")
    url = str(tmp_path / "lancedb")
    ids = [uuid4() for _ in range(30)]
    start = ctx.Event()
    dones = [ctx.Event() for _ in range(4)]
    processes = [ctx.Process(target=_upsert_notes, args=(url, ids, start, done)) for done in dones]
    try:
        for process in processes:
            process.start()
        start.set()
        for done in dones:
            assert done.wait(120)
    finally:
        start.set()
        _join(processes)
    assert [process.exitcode for process in processes] == [0] * len(processes)

    adapter = LanceDBAdapter(url=url, api_key=None, embedding_engine=_HashEmbeddingEngine())

    async def check():
        assert await adapter.count_duplicate_ids("Note_text") == 0
        collection = await adapter.get_collection("Note_text")
        assert await collection.count_rows() == len(ids)

    asyncio.run(check())


def _mutation_waits_for_writer_in_other_process(tmp_path, mutate):
    """Run ``mutate(adapter)`` while another process holds the writer lock and
    check it only completes after that process releases it."""
    ctx = multiprocessing.get_context("spawn")
    url = str(tmp_path / "lancedb")
    attempting, entered, release = ctx.Event(), ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(url, attempting, entered, release))
    Note = _note_type()
    adapter = LanceDBAdapter(url=url, api_key=None, embedding_engine=_HashEmbeddingEngine())

    async def run():
        notes = [Note(text="kept"), Note(text="doomed")]
        await adapter.create_data_points("Note_text", notes)
        holder.start()
        assert await asyncio.to_thread(entered.wait, 30)
        task = asyncio.ensure_future(mutate(adapter, notes))
        await asyncio.sleep(0.5)
        assert not task.done(), "mutation ran while another process held the writer lock"
        release.set()
        return await asyncio.wait_for(task, 30)

    try:
        return asyncio.run(run())
    finally:
        release.set()
        _join([holder])


def test_delete_waits_for_a_writer_in_another_process(tmp_path):
    async def delete(adapter, notes):
        await adapter.delete_data_points("Note_text", [notes[1].id])
        rows = await (await adapter.get_collection("Note_text")).query().to_list()
        return [row["payload"]["text"] for row in rows]

    assert _mutation_waits_for_writer_in_other_process(tmp_path, delete) == ["kept"]


def test_duplicate_repair_waits_for_a_writer_in_another_process(tmp_path):
    import pyarrow as pa

    async def dedupe(adapter, _notes):
        return await adapter.dedupe_collection("Note_text")

    async def seed_and_dedupe(adapter, notes):
        # Seeding writes too, so it queues behind the holder as well.
        collection = await adapter.get_collection("Note_text")
        rows = await collection.query().to_list()
        async with adapter._write_lock():
            await collection.add(pa.Table.from_pylist(rows, schema=await collection.schema()))
        return await dedupe(adapter, notes)

    report = _mutation_waits_for_writer_in_other_process(tmp_path, seed_and_dedupe)
    assert report == {"duplicate_ids": 2, "rows_removed": 2, "ambiguous_ids": []}
