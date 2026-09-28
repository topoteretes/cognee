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
    first_entered, first_release = ctx.Event(), ctx.Event()
    second_attempting, second_entered, second_release = ctx.Event(), ctx.Event(), ctx.Event()
    first = ctx.Process(target=_hold_lock, args=(url, ctx.Event(), first_entered, first_release))
    second = ctx.Process(
        target=_hold_lock,
        args=(url, second_attempting, second_entered, second_release),
    )
    try:
        first.start()
        assert first_entered.wait(10)
        second.start()
        assert second_attempting.wait(10)
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
