"""Unit tests for the Reddit dlt connector.

Two layers, all runnable in CI without live Reddit credentials:

* DB-free tests for rendering (allow-list, whitespace collapse), comment-tree
  truncation (count + depth ceilings, ``more`` chunk expansion), listing
  pagination, and the document-source marker.
* A mocked ``RedditClient`` (httpx MockTransport) for token/401/429 behavior,
  plus dlt-pipeline tests (temp sqlite destination) covering initial load,
  edit re-sync, forget-on-delete, and credential guards.
"""

import httpx
import pytest
from cognee.tasks.ingestion.dlt_utils import document_source_tag

from cognee_community_connector_reddit.reddit import (
    REDDIT_SOURCE_NAME,
    REDDIT_TABLE_POSTS,
    RedditClient,
    _flatten_comments,
    _iter_submissions,
    _one_line,
    _post_to_row,
)


def _post(post_id, title="Post title", author="alice", selftext="Body text", **extra):
    post = {
        "name": f"t3_{post_id}",
        "id": post_id,
        "title": title,
        "author": author,
        "subreddit": "testsub",
        "score": 10,
        "num_comments": 1,
        "created_utc": 1_700_000_000,
        "permalink": f"/r/testsub/comments/{post_id}/t/",
        "selftext": selftext,
    }
    post.update(extra)
    return post


def _comment(comment_id, author, body, score=1, replies=None):
    data = {"author": author, "body": body, "score": score}
    if replies is not None:
        data["replies"] = {"kind": "Listing", "data": {"children": replies}}
    return {"kind": "t1", "data": data, "id": comment_id}


def _more(comment_id, count, children, parent_id):
    return {
        "kind": "more",
        "data": {"count": count, "children": children, "parent_id": parent_id},
        "id": comment_id,
    }


class FakeRedditClient:
    """Stand-in for RedditClient with in-memory fixtures."""

    def __init__(
        self, posts_by_path=None, comment_children=None, more_children=None, page_size=100
    ):
        self._posts = posts_by_path or {}
        self._comment_children = comment_children or {}
        self._more = more_children or {}
        self._page_size = page_size
        self.calls = []

    def get_listing(self, path, *, after=None, limit=100):
        self.calls.append(("listing", path, after))
        posts = self._posts.get(path, [])
        start = 0
        if after:
            for i, post in enumerate(posts):
                if post["name"] == after:
                    start = i + 1
                    break
        return [{"kind": "t3", "data": post} for post in posts[start : start + self._page_size]]

    def get_raw(self, path, *, params=None):
        self.calls.append(("raw", path, params))
        if "/comments/" in path:
            post_id = path.rstrip("/").split("/")[-1]
            return [
                {
                    "kind": "Listing",
                    "data": {"children": self._comment_children.get(post_id, [])},
                }
            ]
        if "morechildren" in path:
            return {"json": {"data": {"things": self._more.get(params["link_id"], [])}}}
        raise AssertionError(f"unexpected raw call {path}")


# ---------------------------------------------------------------------------
# Rendering (DB-free)
# ---------------------------------------------------------------------------


def test_one_line_collapses_whitespace_and_entities():
    assert _one_line("line one\n  line two\t") == "line one line two"
    assert _one_line("a &amp; b") == "a & b"
    assert _one_line(None) == ""
    assert _one_line("x" * 1000) == "x" * 499 + "…"


def test_post_to_row_renders_allowlist():
    row = _post_to_row(
        _post("aaa", title="Hello world", selftext="Interesting body", score=42),
        ["u/bob: nice"],
        permalink="https://www.reddit.com/r/testsub/comments/aaa/t/",
    )
    assert row["id"] == "t3_aaa"
    assert row["title"] == "Hello world"
    assert "u/alice" in row["content"]
    assert "Interesting body" in row["content"]
    assert "u/bob: nice" in row["content"]
    assert "Hello world" not in row["content"]  # title is its own column


def test_post_to_row_omits_non_allowlist_fields():
    post = _post("aaa", media={"reddit_video": {"fallback_url": "https://x/y.mp4"}})
    row = _post_to_row(post, [], permalink="https://www.reddit.com/p")
    assert "media" not in row["content"]
    assert "y.mp4" not in row["content"]
    assert "score" in row["content"].lower()


def test_flatten_respects_count_budget():
    comments = [_comment(f"c{i}", f"u{i}", f"body {i}") for i in range(20)]
    lines = _flatten_comments(
        None,
        comments,
        post_permalink="https://www.reddit.com/t",
        max_count=5,
        max_depth=10,
    )
    assert [line for line in lines if not line.startswith("[more")] == [
        "[+1] u/u0: body 0",
        "[+1] u/u1: body 1",
        "[+1] u/u2: body 2",
        "[+1] u/u3: body 3",
        "[+1] u/u4: body 4",
    ]
    assert lines[-1].startswith("[more comment…")
    assert "all 20 comment threads shown" not in lines


def test_flatten_respects_depth_ceiling():
    a = _comment(
        "c1",
        "a",
        "top",
        replies=[_comment("c2", "b", "middle", replies=[_comment("c3", "c", "deep")])],
    )
    lines = _flatten_comments(
        None,
        [a],
        post_permalink="https://www.reddit.com/t",
        max_count=100,
        max_depth=1,
    )
    assert "deep" not in "\n".join(lines)
    assert "middle" not in "\n".join(lines)
    assert "top" in "\n".join(lines)


def test_flatten_expands_more_chunk():
    more_children = [_comment("m1", "zed", "expanded!")]
    client = FakeRedditClient(more_children={"t3_aaa": more_children})
    root = [_more("mm", count=5, children=["m1"], parent_id="t3_aaa")]
    lines = _flatten_comments(
        client,
        root,
        post_permalink="https://www.reddit.com/t",
        max_count=50,
        max_depth=5,
    )
    assert any("expanded!" in line for line in lines)
    assert any("[more comment…" in line for line in lines)  # still flagged truncated


# ---------------------------------------------------------------------------
# Pagination / source wiring
# ---------------------------------------------------------------------------


def test_iter_submissions_follows_after_cursor():
    posts = [_post(f"p{i}") for i in (1, 2, 3)]
    client = FakeRedditClient(posts_by_path={"/r/ts/hot": posts}, page_size=1)
    collected = list(_iter_submissions(client, "ts", "hot", limit=10))
    assert [p["name"] for p in collected] == ["t3_p1", "t3_p2", "t3_p3"]
    assert client.calls[0] == ("listing", "/r/ts/hot", None)
    assert client.calls[1][2] == "t3_p1"
    assert client.calls[2][2] == "t3_p2"


def test_iter_submissions_respects_limit():
    posts = [_post(f"p{i}") for i in range(5)]
    client = FakeRedditClient(posts_by_path={"/r/ts/hot": posts})
    collected = list(_iter_submissions(client, "ts", "hot", limit=2))
    assert len(collected) == 2


def test_reddit_source_declares_document_marker():
    from cognee_community_connector_reddit.reddit import reddit_source

    source = reddit_source(client=FakeRedditClient())
    assert REDDIT_SOURCE_NAME == "reddit"
    assert document_source_tag(source) == "reddit"


def test_reddit_source_requires_credentials(monkeypatch):
    from cognee_community_connector_reddit.reddit import reddit_source

    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USERNAME", "REDDIT_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="credentials"):
        reddit_source()


def test_reddit_source_reads_credentials_from_env(monkeypatch):
    from cognee_community_connector_reddit.reddit import reddit_source

    monkeypatch.setenv("REDDIT_CLIENT_ID", "demo-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "demo-secret")
    monkeypatch.setenv("REDDIT_USERNAME", "demo-user")
    monkeypatch.setenv("REDDIT_PASSWORD", "demo-pass")
    source = reddit_source(subreddits=["ts"])
    assert source.name == REDDIT_SOURCE_NAME
    assert set(source.resources) == {REDDIT_TABLE_POSTS}


def _run_sync(dlt, tmp_path, fake_client, **kwargs):
    from cognee_community_connector_reddit.reddit import reddit_source

    db_path = (tmp_path / "reddit.db").as_posix()
    pipeline = dlt.pipeline(
        pipeline_name="reddit_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{db_path}"),
        dataset_name="reddit_ds",
        pipelines_dir=str(tmp_path / "state"),
    )
    pipeline.run(reddit_source(client=fake_client, subreddits=["ts"], **kwargs))
    return pipeline


def _read_table(pipeline, table):
    with pipeline.sql_client() as client:
        rows = client.execute_sql(f"SELECT id, title, content FROM {table}")
    return {row[0]: {"id": row[0], "title": row[1], "content": row[2]} for row in rows}


@pytest.fixture
def dlt_mod():
    return pytest.importorskip("dlt")


def test_first_sync_loads_posts(dlt_mod, tmp_path):
    client = FakeRedditClient(
        posts_by_path={"/r/ts/hot": [_post("aaa", title="First")]},
        comment_children={"aaa": [_comment("c1", "bob", "a comment")]},
    )
    pipeline = _run_sync(dlt_mod, tmp_path, client)
    rows = _read_table(pipeline, REDDIT_TABLE_POSTS)
    assert set(rows) == {"t3_aaa"}
    assert "a comment" in rows["t3_aaa"]["content"]


def test_edit_is_reflected_on_resync(dlt_mod, tmp_path):
    client = FakeRedditClient(posts_by_path={"/r/ts/hot": [_post("aaa", title="First")]})
    _run_sync(dlt_mod, tmp_path, client)

    edited = FakeRedditClient(posts_by_path={"/r/ts/hot": [_post("aaa", title="Edited")]})
    pipeline = _run_sync(dlt_mod, tmp_path, edited)

    rows = _read_table(pipeline, REDDIT_TABLE_POSTS)
    assert rows["t3_aaa"]["title"] == "Edited"


def test_vanished_post_is_removed_on_resync(dlt_mod, tmp_path):
    client = FakeRedditClient(posts_by_path={"/r/ts/hot": [_post("aaa"), _post("bbb")]})
    _run_sync(dlt_mod, tmp_path, client)

    vanished = FakeRedditClient(posts_by_path={"/r/ts/hot": [_post("bbb")]})
    pipeline = _run_sync(dlt_mod, tmp_path, vanished)

    rows = _read_table(pipeline, REDDIT_TABLE_POSTS)
    assert "t3_aaa" not in rows
    assert "t3_bbb" in rows


# ---------------------------------------------------------------------------
# HTTP client behavior (mocked transport)
# ---------------------------------------------------------------------------


def test_client_retries_429_with_retry_after():
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "tok1"})
        state["attempts"] += 1
        if state["attempts"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "kind": "Listing",
                "data": {"children": [{"kind": "t3", "data": {"name": "t3_a"}}]},
            },
        )

    client = RedditClient("id", "secret", "user", "pass", transport=httpx.MockTransport(handler))
    children = client.get_listing("/r/ts/hot")
    assert children[0]["data"]["name"] == "t3_a"


def test_client_reauths_once_on_401():
    state = {"tokens": ["tok1", "tok2"], "tok": 0, "rejected": False}
    token_index = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/access_token":
            token_index["n"] += 1
            return httpx.Response(200, json={"access_token": f"tok{token_index['n']}"})
        if request.headers.get("Authorization") == "Bearer tok1" and not state["rejected"]:
            state["rejected"] = True
            return httpx.Response(401, json={"message": "Unauthorized"})
        return httpx.Response(
            200,
            json={
                "kind": "Listing",
                "data": {"children": [{"kind": "t3", "data": {"name": "t3_b"}}]},
            },
        )

    client = RedditClient("id", "secret", "user", "pass", transport=httpx.MockTransport(handler))
    children = client.get_listing("/r/ts/hot")
    assert children[0]["data"]["name"] == "t3_b"
    assert token_index["n"] == 2  # initial token + one refresh


def test_client_gives_up_after_exhausting_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "tok1"})
        return httpx.Response(429, headers={"Retry-After": "0"})

    from cognee_community_connector_reddit.reddit import RedditAPIError

    client = RedditClient(
        "id",
        "secret",
        "user",
        "pass",
        transport=httpx.MockTransport(handler),
        max_retries=2,
    )
    with pytest.raises(RedditAPIError):
        client.get_listing("/r/ts/hot")
