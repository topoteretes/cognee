"""SSE transport for the streamed graph read.

What has to hold. The events arrive in their documented order, and no link
refers to a node that has not been sent. A failure before the first chunk is
raised, so the route can still give it a status code; a failure after it is one
error event. The read never waits for the client, so the dataset slot it holds
is released as soon as the store has answered, whatever the client does. A
client that goes away stops the read. And the process never holds more streams,
or holds one for longer, than it allows.
"""

import asyncio
import json

import pytest

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.graph.bounded_neighborhood import hop_distances
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.visualization import graph_stream
from cognee.modules.visualization.graph_stream import begin_graph_stream, stream_graph_events


@pytest.fixture(autouse=True)
def fresh_permits(monkeypatch):
    """A fresh permit pool per test, checked full again at the end: a stream
    that leaks its permit is a stream nobody can open once the pool is gone."""
    monkeypatch.setattr(graph_stream, "_permits", None)
    yield
    permits = graph_stream._permits
    if permits is not None:
        assert permits._value == graph_stream.MAX_STREAMS_IN_FLIGHT, "a permit was not returned"


class _Engine:
    """An adapter without a native bounded read, so the interface default runs."""

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        distance = hop_distances(self._edges, node_ids)
        reached = {node_id for node_id, hops in distance.items() if hops <= depth}
        nodes = [node for node in self._nodes if node[0] in reached]
        edges = [edge for edge in self._edges if edge[0] in reached and edge[1] in reached]
        return nodes, edges

    def iter_bounded_neighborhood(self, *args, **kwargs):
        return GraphDBInterface.iter_bounded_neighborhood(self, *args, **kwargs)

    async def get_graph_data(self):
        # Degree seeding's inherited default, reached when no seed is given.
        return self._nodes, self._edges


def _star(spokes: int = 25):
    nodes = [("hub", {"name": "hub", "type": "Entity"})]
    nodes += [
        (f"c{index}", {"type": "DocumentChunk", "text": f"chunk {index} " * 40})
        for index in range(spokes)
    ]
    edges = [("hub", f"c{index}", "contains", {}) for index in range(spokes)]
    return nodes, edges


def _events(engine, **overrides):
    options = {
        "query": None,
        "seed_node_ids": ["hub"],
        "neighborhood_depth": 1,
        "seed_top_k": 10,
        "max_nodes": 100,
        "chunk_size": 10,
    }
    options.update(overrides)
    return stream_graph_events(engine, **options)


def _parse(frames: list[str]) -> list[tuple[str, dict]]:
    events = []
    for frame in frames:
        if frame.startswith(":"):
            continue
        lines = frame.strip().splitlines()
        event = lines[0].removeprefix("event: ")
        events.append((event, json.loads(lines[1].removeprefix("data: "))))
    return events


async def _collect(stream) -> list[str]:
    return [frame async for frame in stream.frames()]


@pytest.mark.asyncio
async def test_events_arrive_in_order_and_links_never_run_ahead_of_nodes():
    stream = await begin_graph_stream(_events(_Engine(*_star())))
    events = _parse(await _collect(stream))

    names = [event for event, _ in events]
    assert names == ["meta", "chunk", "chunk", "chunk", "summary", "summary", "summary", "done"]
    assert events[0][1]["seeds"] == ["hub"]
    assert events[0][1]["seed_source"] == "explicit"

    sent = set()
    for event, data in events:
        if event != "chunk":
            continue
        assert len(data["nodes"]) <= 10
        sent.update(node["id"] for node in data["nodes"])
        for link in data["links"]:
            assert link["source"] in sent and link["target"] in sent
        for node in data["nodes"]:
            assert "text" not in node  # the compact shape, not the property bag

    done = events[-1][1]
    assert done == {"nodes": 26, "links": 25, "chunks": 3}
    summaries = [data for event, data in events if event == "summary"]
    assert all(len(part["nodes"]) <= 10 for part in summaries)
    assert "color_maps" in summaries[0]
    assert all("color_maps" not in part for part in summaries[1:])
    merged = {node_id: value for part in summaries for node_id, value in part["nodes"].items()}
    assert set(merged) == sent
    assert merged["hub"]["importance"] == 1.0


@pytest.mark.asyncio
async def test_chunk_nodes_are_named_from_their_text():
    stream = await begin_graph_stream(_events(_Engine(*_star(spokes=3))))
    events = _parse(await _collect(stream))

    names = {
        node["id"]: node["name"]
        for event, data in events
        if event == "chunk"
        for node in data["nodes"]
    }
    assert names["c1"].startswith("chunk 1 chunk 1")
    assert len(names["c1"]) == 120


@pytest.mark.asyncio
async def test_no_seeds_is_an_empty_graph_not_an_error():
    stream = await begin_graph_stream(_events(_Engine([], []), seed_node_ids=None))
    events = _parse(await _collect(stream))

    assert [event for event, _ in events] == ["meta", "summary", "done"]
    assert events[-1][1] == {"nodes": 0, "links": 0, "chunks": 0}


async def _failing_before_any_chunk():
    raise ValueError("vector store unreachable")
    yield  # pragma: no cover


@pytest.mark.asyncio
async def test_a_failure_before_the_first_chunk_is_raised():
    """The route can then answer it with a status code, not a 200 and an event."""
    with pytest.raises(ValueError, match="vector store unreachable"):
        await begin_graph_stream(_failing_before_any_chunk())


async def _failing_after_one_chunk(error):
    yield "meta", {"seeds": ["a"]}
    yield "chunk", {"index": 0, "nodes": [{"id": "a"}], "links": []}
    raise error


class _Denied(CogneeApiError):
    def __init__(self):
        super().__init__(message="No access to dataset", status_code=403)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_Denied(), {"message": "No access to dataset", "status": 403}),
        (
            RuntimeError("password=hunter2 in the DSN"),
            {"message": "Failed to build the visualization payload", "status": 409},
        ),
    ],
)
async def test_a_failure_after_the_first_chunk_is_one_error_event(error, expected):
    stream = await begin_graph_stream(_failing_after_one_chunk(error))
    events = _parse(await _collect(stream))

    assert [event for event, _ in events] == ["meta", "chunk", "error"]
    assert events[-1][1] == expected  # the status the JSON path would give, no detail


@pytest.mark.asyncio
async def test_a_disconnect_cancels_the_read_and_unwinds_it():
    unwound = asyncio.Event()

    async def endless():
        try:
            yield "meta", {}
            index = 0
            while True:
                yield "chunk", {"index": index, "nodes": [], "links": []}
                index += 1
        finally:
            unwound.set()  # where the dataset context and the session close

    stream = await begin_graph_stream(endless())
    frames = stream.frames()
    await frames.__anext__()
    await frames.__anext__()
    await frames.aclose()  # what Starlette does when the client goes away

    assert unwound.is_set()
    assert stream._task.done()


@pytest.mark.asyncio
async def test_the_read_runs_to_its_end_without_waiting_for_the_client():
    produced = []

    async def counting():
        for index in range(50):
            produced.append(index)
            yield "chunk", {"index": index, "nodes": [], "links": []}

    stream = await begin_graph_stream(counting())
    await asyncio.wait_for(asyncio.shield(stream._task), timeout=2)  # an idle client

    assert len(produced) == 50
    frames = [frame async for frame in stream.frames()]
    assert sum(frame.startswith("event: chunk") for frame in frames) == 50


@pytest.mark.asyncio
async def test_the_read_lets_the_event_loop_run_between_chunks():
    """The interface default yields chunks from memory with no await between
    them. Without a yield per frame, the read would encode the whole graph
    while every other request on the process waited."""
    loop_ran = asyncio.Event()
    ran_before_the_read_ended = []

    async def in_memory():
        asyncio.get_running_loop().call_soon(loop_ran.set)
        for index in range(3):
            yield "chunk", {"index": index, "nodes": [], "links": []}
        ran_before_the_read_ended.append(loop_ran.is_set())

    stream = await begin_graph_stream(in_memory())
    await _collect(stream)

    assert ran_before_the_read_ended == [True]


@pytest.mark.asyncio
async def test_a_silent_read_sends_keepalives(monkeypatch):
    monkeypatch.setattr(graph_stream, "KEEPALIVE_SECONDS", 0.01)
    release = asyncio.Event()

    async def slow():
        yield "chunk", {"index": 0, "nodes": [], "links": []}
        await release.wait()
        yield "done", {}

    stream = await begin_graph_stream(slow())
    frames = stream.frames()
    first = await frames.__anext__()
    second = await frames.__anext__()
    release.set()
    rest = [frame async for frame in frames]

    assert first.startswith("event: chunk")
    assert second == ": keepalive\n\n"
    assert any(frame.startswith("event: done") for frame in rest)


@pytest.mark.asyncio
async def test_a_stream_nobody_reads_gives_its_permit_back_after_its_lifetime(monkeypatch):
    """A stalled client, or a body the server never iterated, keeps no permit
    and no frames past the lifetime."""
    monkeypatch.setattr(graph_stream, "STREAM_LIFETIME_SECONDS", 0.05)

    async def graph():
        yield "chunk", {"index": 0, "nodes": [], "links": []}
        yield "done", {}

    stream = await begin_graph_stream(graph())
    assert graph_stream._stream_permits()._value == graph_stream.MAX_STREAMS_IN_FLIGHT - 1
    await asyncio.sleep(0.1)

    assert graph_stream._stream_permits()._value == graph_stream.MAX_STREAMS_IN_FLIGHT
    assert stream._buffered == []
    # A client that does come back is told, not left with a stream cut short.
    events = _parse([frame async for frame in stream.frames()])
    assert events == [
        (
            "error",
            {"message": "The graph stream was not read in time and was closed.", "status": 408},
        )
    ]


@pytest.mark.asyncio
async def test_a_stream_read_in_time_is_not_cut_by_its_lifetime(monkeypatch):
    monkeypatch.setattr(graph_stream, "STREAM_LIFETIME_SECONDS", 0.05)
    stream = await begin_graph_stream(_events(_Engine(*_star())))
    events = _parse(await _collect(stream))
    await asyncio.sleep(0.1)  # past the lifetime, after the body finished

    assert events[-1][0] == "done"


# --- the permit pool ----------------------------------------------------------


async def _one_chunk():
    yield "chunk", {"index": 0, "nodes": [], "links": []}
    yield "done", {}


@pytest.mark.asyncio
async def test_streams_beyond_the_cap_are_refused_before_anything_is_read(monkeypatch):
    from cognee.modules.visualization.exceptions import GraphStreamCapacityError

    monkeypatch.setattr(graph_stream, "MAX_STREAMS_IN_FLIGHT", 2)
    opened = [await begin_graph_stream(_one_chunk()) for _ in range(2)]
    started = []

    async def must_not_start():
        started.append(True)
        yield "chunk", {}

    with pytest.raises(GraphStreamCapacityError) as refused:
        await begin_graph_stream(must_not_start())
    assert refused.value.status_code == 503
    assert started == []

    await _collect(opened[0])  # one finishes, its permit comes back
    replacement = await begin_graph_stream(_one_chunk())
    for stream in (opened[1], replacement):
        await _collect(stream)


async def _fails_before_a_chunk():
    raise ValueError("nope")
    yield  # pragma: no cover


async def _fails_after_a_chunk():
    yield "chunk", {"index": 0, "nodes": [], "links": []}
    raise ValueError("nope")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending",
    ["completes", "fails_before_a_chunk", "fails_after_a_chunk", "client_disconnects", "closed"],
)
async def test_every_ending_gives_the_permit_back(ending):
    """Checked by the fresh_permits fixture after the test."""
    if ending == "fails_before_a_chunk":
        with pytest.raises(ValueError):
            await begin_graph_stream(_fails_before_a_chunk())
        return
    events = _fails_after_a_chunk() if ending == "fails_after_a_chunk" else _one_chunk()
    stream = await begin_graph_stream(events)
    if ending == "client_disconnects":
        frames = stream.frames()
        await frames.__anext__()
        await frames.aclose()
    elif ending == "closed":
        await stream.close()  # the route's BackgroundTask, body never iterated
        await stream.close()  # and again, as frames() would
    else:
        await _collect(stream)


@pytest.mark.asyncio
async def test_a_request_cancelled_before_its_headers_gives_the_permit_back():
    """The client went away while the first chunk was still being read."""
    gate = asyncio.Event()

    async def slow_selection():
        await gate.wait()
        yield "chunk", {}

    begin = asyncio.create_task(begin_graph_stream(slow_selection()))
    await asyncio.sleep(0.01)
    begin.cancel()
    with pytest.raises(asyncio.CancelledError):
        await begin


# --- encoding ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_value_json_cannot_encode_is_an_error_event_not_a_cut_stream():
    from uuid import uuid4

    async def graph():
        yield "chunk", {"index": 0, "nodes": [{"id": "a"}], "links": []}
        yield "chunk", {"index": 1, "nodes": [{"id": "b", "source_node_set": uuid4()}]}

    stream = await begin_graph_stream(graph())
    events = _parse(await _collect(stream))

    assert [event for event, _ in events] == ["chunk", "error"]
    assert events[-1][1]["status"] == 409


@pytest.mark.asyncio
async def test_a_value_json_cannot_encode_before_the_first_chunk_is_raised():
    from uuid import uuid4

    async def graph():
        yield "meta", {"seeds": [uuid4()]}

    with pytest.raises(TypeError):
        await begin_graph_stream(graph())


@pytest.mark.asyncio
async def test_a_failure_after_meta_but_before_a_chunk_is_still_raised():
    """Seeds resolve, then the membership query fails: still before any byte."""

    async def fails_in_selection():
        yield "meta", {"seeds": ["a"]}
        raise ValueError("membership query failed")

    with pytest.raises(ValueError, match="membership query failed"):
        await begin_graph_stream(fails_in_selection())


def _recording_context(log):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def recording_context(*_args, **_kwargs):
        log.append(("enter", asyncio.current_task()))
        try:
            yield
        finally:
            log.append(("exit", asyncio.current_task()))

    return recording_context


@pytest.mark.asyncio
async def test_the_dataset_context_is_left_when_the_read_ends_not_when_the_client_does(
    monkeypatch,
):
    """The context's queue slot belongs to the task that entered it, so the
    read enters and leaves it in its own task; and it leaves as soon as the
    store has answered, while the client has not read past the first chunk."""
    import sys
    from types import SimpleNamespace

    from cognee.api.v1.visualize.visualize import stream_dataset_graph

    visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]
    log = []

    async def engine():
        return _Engine(*_star(spokes=15))

    monkeypatch.setattr(
        visualize_module, "set_database_global_context_variables", _recording_context(log)
    )
    monkeypatch.setattr(visualize_module, "get_graph_engine", engine)

    dataset = SimpleNamespace(id="d", owner_id="o")
    stream = await begin_graph_stream(
        stream_dataset_graph(dataset, seed_node_ids=["hub"], neighborhood_depth=1, chunk_size=5)
    )
    await asyncio.wait_for(asyncio.shield(stream._task), timeout=2)

    assert [event for event, _ in log] == ["enter", "exit"]
    assert log[0][1] is log[1][1] is stream._task
    await stream.close()


@pytest.mark.asyncio
async def test_slow_clients_do_not_hold_dataset_slots(monkeypatch):
    """The confirmed review finding, with the real DatasetQueue: two streams
    on one dataset whose clients read slowly used to hold both slots of a
    two-slot queue until they finished, and an operation on another dataset
    waited for all of it."""
    import sys
    import time
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from cognee.api.v1.visualize.visualize import stream_dataset_graph
    from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

    visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]
    queue = DatasetQueue(enabled=True, max_concurrent=2, idle_ttl_seconds=0)
    queue._release_subprocess_engines = lambda: None

    @asynccontextmanager
    async def queued_context(dataset_id, *_args, **_kwargs):
        await queue.ensure_slot(dataset_id)
        try:
            yield
        finally:
            await queue.release_slot_for(dataset_id)

    async def engine():
        return _Engine(*_star(spokes=40))

    monkeypatch.setattr(visualize_module, "set_database_global_context_variables", queued_context)
    monkeypatch.setattr(visualize_module, "get_graph_engine", engine)

    dataset = SimpleNamespace(id="same", owner_id="o")
    streams = [
        await begin_graph_stream(
            stream_dataset_graph(dataset, seed_node_ids=["hub"], neighborhood_depth=1, chunk_size=4)
        )
        for _ in range(2)
    ]

    async def slow_client(stream):
        async for _ in stream.frames():
            await asyncio.sleep(0.02)

    clients = [asyncio.create_task(slow_client(stream)) for stream in streams]
    started = time.monotonic()
    await queue.ensure_slot("other")
    waited = time.monotonic() - started
    await queue.release_slot_for("other")
    await asyncio.gather(*clients)

    assert waited < 0.1, f"another dataset waited {waited:.2f} s behind slow clients"


class _TypedEngine(_Engine):
    """alice is_a person, bob is_a person; the type lookup always fails."""

    def __init__(self):
        super().__init__(
            [
                ("alice", {"name": "Alice", "type": "Entity"}),
                ("person", {"name": "Person", "type": "EntityType"}),
                ("bob", {"name": "Bob", "type": "Entity"}),
            ],
            [("alice", "person", "is_a", {}), ("bob", "person", "is_a", {})],
        )
        self.lookups = 0

    async def get_entity_type_names(self, entity_ids):
        self.lookups += 1
        raise RuntimeError("no native type lookup")


@pytest.mark.asyncio
async def test_after_a_failed_lookup_the_summary_types_entities_like_the_json_path():
    """SDK-794: alice is sent before her type node; the summary corrects her."""
    from cognee.modules.visualization.preprocessor import preprocess
    from cognee.modules.visualization.subgraph_data import fetch_visualization_graph_data

    engine = _TypedEngine()
    events = [event async for event in _events(engine, seed_node_ids=["alice"], chunk_size=1)]

    streamed = {}
    for name, data in events:
        if name == "chunk":
            streamed.update({node["id"]: node["entity_type"] for node in data["nodes"]})
    assert streamed["alice"] == "Entity"  # her is_a link came in a later chunk
    for name, data in events:
        if name == "summary":
            for node_id, entry in data["nodes"].items():
                streamed[node_id] = entry.get("entity_type", streamed[node_id])

    graph_data = await fetch_visualization_graph_data(
        engine, seed_node_ids=["alice"], neighborhood_depth=1
    )
    json_types = {node["id"]: node["entity_type"] for node in preprocess(graph_data).nodes}
    assert streamed == json_types == {"alice": "Person", "person": "EntityType"}


@pytest.mark.asyncio
async def test_a_failed_type_lookup_is_not_retried_for_every_chunk():
    engine = _TypedEngine()
    # Depth 2 puts alice and bob, both entities, in separate chunks.
    events = [
        event
        async for event in _events(
            engine, seed_node_ids=["alice"], neighborhood_depth=2, chunk_size=1
        )
    ]
    assert sum(name == "chunk" for name, _ in events) == 3
    assert engine.lookups == 1


@pytest.mark.asyncio
async def test_the_summary_adds_entity_type_only_where_it_corrects_a_chunk():
    events = [event async for event in _events(_Engine(*_star()))]
    for name, data in events:
        if name == "summary":
            assert all("entity_type" not in entry for entry in data["nodes"].values())
