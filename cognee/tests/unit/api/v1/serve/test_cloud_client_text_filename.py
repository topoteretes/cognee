"""Raw-text cloud uploads must be named by content hash, not a shared placeholder.

A fixed ``data.txt`` made every text upload for a tenant land on one remote
object; rapid adds then raced the server's content-hash read-back into
FileContentHashingError 409s (first surfaced by the nightly cloud 50-small-
documents benchmark). The name must mirror local ingestion's
``save_data_to_file`` convention (``text_<md5>.txt``) so identical content
dedupes to the same key and distinct content never collides.
"""

import hashlib

from cognee.api.v1.serve.cloud_client import _text_upload_filename
from cognee.modules.ingestion.data_types.TextData import create_text_data


def test_filename_is_content_hash_in_local_ingestion_format():
    text = "a small memory"
    expected_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    assert _text_upload_filename(text) == f"text_{expected_hash}.txt"


def test_filename_matches_local_ingestion_namer_exactly():
    # The contract: cloud uploads and save_data_to_file use ONE naming source
    # (TextData), so remote and local names can never drift apart.
    text = "shared naming contract"
    assert _text_upload_filename(text) == create_text_data(text).get_metadata()["name"]


def test_distinct_texts_get_distinct_filenames():
    names = {_text_upload_filename(f"memory number {i}") for i in range(50)}
    assert len(names) == 50


def test_identical_texts_share_a_filename():
    assert _text_upload_filename("same") == _text_upload_filename("same")


def test_unicode_text_hashes_utf8_bytes():
    text = "café ☕ разговор"
    expected_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    assert _text_upload_filename(text) == f"text_{expected_hash}.txt"
