# Adapted from topoteretes/cognee-community (Apache-2.0), PR #316, commit a154bd4.
# Modified to ship in the Cognee SDK: serves Dropbox-shaped JSON over httpx.MockTransport.
"""An in-memory Dropbox behind ``httpx.MockTransport``.

It answers the routes the connector calls with the JSON shapes of the Dropbox API reference
(``files/get_metadata``, ``files/list_folder``, ``files/list_folder/continue``,
``files/download``, ``files/export``, ``users/get_current_account``), so the real
``DropboxClient`` runs under test. Every change made through the helpers is appended to a
change log, which is what a ``list_folder/continue`` cursor replays: a move is a delete of the
old path plus the same id at the new path, a deleted folder is one entry.
"""

import hashlib
import json
import posixpath

import httpx

TOKEN = "sl.fake-token-0123456789"
DEFAULT_MODIFIED = "2026-01-01T09:00:00Z"
ACCOUNT = "dbid:AAAfakeaccount"


def lookup_error(tag: str, outer: str = "path") -> tuple[int, dict]:
    """A 409 body like ``{"error": {".tag": "path", "path": {".tag": "not_found"}}}``."""
    if outer is None:
        return 409, {"error_summary": f"{tag}/..", "error": {".tag": tag}}
    return 409, {
        "error_summary": f"{outer}/{tag}/..",
        "error": {".tag": outer, outer: {".tag": tag}},
    }


class FakeDropbox:
    def __init__(self, page_size: int = 2000, *, streamed: bool = True):
        """``streamed`` answers with unread streaming bodies, as a real transport does, so a
        client that reads a body it never streamed in fails here as it does in production."""
        self.streamed = streamed
        self.page_size = page_size
        self.files: dict[str, dict] = {}  # path_lower -> file metadata
        self.folders: dict[str, dict] = {}  # path_lower -> folder metadata
        self.content: dict[str, bytes] = {}  # file id -> bytes
        self.log: list[dict] = []
        self.reset_cursors: set[str] = set()
        # One answer each, in order, for the next request to that route: (status, body, headers).
        self.queued: dict[str, list[tuple]] = {}
        # Every download or export of that file id answers this, until removed.
        self.download_errors: dict[str, tuple] = {}
        self.calls: list[str] = []
        self.downloads: list[str] = []
        self.exports: list[tuple[str, str]] = []
        self.authorizations: list[str] = []
        self.account = ACCOUNT
        self.team_space = False
        self._pages: dict[str, tuple[list, str]] = {}
        self._ids = 0

    # ------------------------------------------------------------------ changes
    def put(
        self,
        path,
        content=b"",
        *,
        downloadable=True,
        export_info=None,
        size=None,
        modified=DEFAULT_MODIFIED,
        server_modified=None,
    ) -> str:
        """Create or edit a file. ``modified`` is its ``client_modified``; ``None`` leaves it
        out. ``server_modified`` defaults to the same time."""
        data = content.encode() if isinstance(content, str) else content
        self._ensure_folder(posixpath.dirname(path))
        existing = self.files.get(path.lower())
        file_id = existing["id"] if existing else self._new_id("file")
        meta = self._file(path, file_id, size if size is not None else len(data))
        if modified is not None:
            meta["client_modified"] = modified
        if server_modified or modified:
            meta["server_modified"] = server_modified or modified
        if not downloadable:
            meta["is_downloadable"] = False
            meta["export_info"] = export_info
        self.files[path.lower()] = meta
        self.content[file_id] = data
        # Dropbox's content_hash is a block hash of the bytes; the fake only needs it to be
        # the same for the same bytes and different otherwise.
        meta["content_hash"] = hashlib.sha256(data).hexdigest()
        if not downloadable:
            # A Paper doc carries no content_hash: it has no bytes of its own to download.
            del meta["content_hash"]
        self.log.append(meta)
        return file_id

    def mkdir(self, path: str) -> str:
        self._ensure_folder(path)
        return self.folders[path.lower()]["id"]

    def move(self, old: str, new: str) -> None:
        """Move or rename a file or a folder (with everything under it)."""
        lower, new_lower = old.lower(), new.lower()
        self.log.append(self._deleted(old))
        if lower in self.files:
            meta = self.files.pop(lower)
            self._ensure_folder(posixpath.dirname(new))
            moved = {**meta, **self._paths(new)}
            self.files[new_lower] = moved
            self.log.append(moved)
            return
        self._ensure_folder(posixpath.dirname(new))
        for store in (self.folders, self.files):
            for key in sorted(k for k in store if k == lower or k.startswith(lower + "/")):
                meta = store.pop(key)
                display = new + meta["path_display"][len(old) :]
                moved = {**meta, **self._paths(display)}
                store[display.lower()] = moved
                self.log.append(moved)

    def delete(self, path: str) -> None:
        lower = path.lower()
        for store in (self.files, self.folders):
            for key in [k for k in store if k == lower or k.startswith(lower + "/")]:
                del store[key]
        self.log.append(self._deleted(path))

    def vanish(self, path: str) -> None:
        """Remove a file without a delete in the log (Dropbox collapses delete + re-create)."""
        self.files.pop(path.lower())

    # ------------------------------------------------------------------ transport
    def client(self, token=TOKEN, **kwargs):
        from cognee.tasks.ingestion.connectors.dropbox import DropboxClient

        return DropboxClient(
            token, http=httpx.Client(transport=httpx.MockTransport(self.handle)), **kwargs
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        response = self._handle(request)
        if not self.streamed:
            return response
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=httpx.ByteStream(response.content),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        route = request.url.path.removeprefix("/2/")
        self.calls.append(route)
        self.authorizations.append(request.headers.get("Authorization", ""))
        if self.queued.get(route):
            status, body, headers = (*self.queued[route].pop(0), {})[:3]
            return self._answer(status, body, headers)
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._answer(401, {"error": {".tag": "invalid_access_token"}})
        if request.url.host == "content.dropboxapi.com":
            arg = json.loads(request.headers["Dropbox-API-Arg"])
            return self._content(route, arg)
        body = json.loads(request.content or b"null")
        return getattr(self, "_" + route.replace("/", "_"))(body)

    def _answer(self, status, body, headers=None) -> httpx.Response:
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, headers=headers or {})
        return httpx.Response(status, json=body, headers=headers or {})

    def _users_get_current_account(self, _body):
        return self._answer(
            200,
            {
                "account_id": self.account,
                "email": "user@example.com",
                "root_info": {
                    ".tag": "team" if self.team_space else "user",
                    "root_namespace_id": "100" if self.team_space else "200",
                    "home_namespace_id": "200",
                },
            },
        )

    def _files_get_metadata(self, body):
        target = body["path"]
        for store in (self.files, self.folders):
            for meta in store.values():
                if meta["id"] == target or meta["path_lower"] == target.lower():
                    return self._answer(200, meta)
        return self._answer(*lookup_error("not_found"))

    def _files_list_folder(self, body):
        assert body["recursive"] is True, "the connector always lists recursively"
        assert body["limit"] == 2000, "the documented maximum page size"
        assert body["include_non_downloadable_files"] is True, "Paper docs are listed too"
        path = self._folder_path(body["path"])
        if path is None:
            return self._answer(*lookup_error("not_found"))
        entries = [
            meta
            for store in (self.folders, self.files)
            for key, meta in sorted(store.items())
            if self._under(key, path)
        ]
        return self._page(entries, self._cursor(body["path"]))

    def _files_list_folder_continue(self, body):
        cursor = body["cursor"]
        if cursor in self._pages:
            entries, final = self._pages.pop(cursor)
            return self._page(entries, final)
        if cursor in self.reset_cursors:
            return self._answer(*lookup_error("reset", None))
        target, position = cursor.rsplit("|", 1)
        path = self._folder_path(target)
        if path is None:
            return self._answer(*lookup_error("not_found"))
        entries = [
            entry
            for entry in self.log[int(position) :]
            # An unmounted entry carries no path, and is reported to every listing.
            if "path_lower" not in entry or self._under(entry["path_lower"], path)
        ]
        return self._page(entries, self._cursor(target))

    def _content(self, route, arg):
        file_id = arg["path"]
        if route == "files/download":
            self.downloads.append(file_id)
        else:
            self.exports.append((file_id, arg["export_format"]))
        if file_id in self.download_errors:
            status, body, headers = (*self.download_errors[file_id], {})[:3]
            return self._answer(status, body, headers)
        meta = next((m for m in self.files.values() if m["id"] == file_id), None)
        if meta is None:
            return self._answer(*lookup_error("not_found"))
        if route == "files/download" and not meta["is_downloadable"]:
            # Dropbox refuses to download a Paper doc: it can only be exported.
            return self._answer(*lookup_error("unsupported_file", None))
        if route == "files/export" and meta["is_downloadable"]:
            return self._answer(*lookup_error("non_exportable", None))
        return httpx.Response(
            200,
            content=self.content[file_id],
            headers={"Dropbox-API-Result": json.dumps({"id": file_id})},
        )

    # ------------------------------------------------------------------ internals
    def _folder_path(self, target: str) -> str | None:
        if target == "":
            return ""
        for meta in self.folders.values():
            if meta["id"] == target or meta["path_lower"] == target.lower():
                return meta["path_lower"]
        return None

    def _page(self, entries, final):
        if len(entries) <= self.page_size:
            return self._answer(200, {"entries": entries, "cursor": final, "has_more": False})
        token = f"page-{len(self._pages)}-{len(self.log)}-{len(entries)}"
        self._pages[token] = (entries[self.page_size :], final)
        return self._answer(
            200, {"entries": entries[: self.page_size], "cursor": token, "has_more": True}
        )

    def _cursor(self, target: str) -> str:
        return f"{target}|{len(self.log)}"

    def _ensure_folder(self, path: str) -> None:
        if path in ("", "/") or path.lower() in self.folders:
            return
        self._ensure_folder(posixpath.dirname(path))
        meta = {".tag": "folder", "id": self._new_id("folder"), **self._paths(path)}
        self.folders[path.lower()] = meta
        self.log.append(meta)

    @staticmethod
    def _paths(path: str) -> dict:
        return {"name": posixpath.basename(path), "path_lower": path.lower(), "path_display": path}

    def _file(self, path, file_id, size):
        self._ids += 1
        return {
            ".tag": "file",
            "id": file_id,
            **self._paths(path),
            "size": size,
            "rev": f"{self._ids:016x}",
            "is_downloadable": True,
        }

    def _deleted(self, path):
        return {".tag": "deleted", **self._paths(path)}

    def _new_id(self, kind):
        self._ids += 1
        return f"id:{kind}{self._ids}"

    @staticmethod
    def _under(path: str, folder: str) -> bool:
        return folder == "" or path == folder or path.startswith(folder + "/")


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF whose text pypdf can extract."""
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)
