from pathlib import Path

import pytest

from cognee.modules.presort import local_paths


def test_default_roots_are_bounded(monkeypatch, tmp_path):
    monkeypatch.delenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", raising=False)
    roots = local_paths.get_presort_roots()
    assert Path.cwd().resolve() in roots
    assert Path(local_paths.get_base_config().system_root_directory).resolve() in roots
    assert local_paths.resolve_presort_path(tmp_path) == tmp_path.resolve()
    # A home directory is outside cwd, temp, and Cognee's storage directories.
    # Use an absolute root that is not inside any permitted directory.
    outside = Path(tmp_path.anchor) / "unapproved-presort-source"
    with pytest.raises(ValueError, match="outside allowed roots"):
        local_paths.resolve_presort_path(outside)


def test_configured_roots_override_defaults(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", str(allowed))
    assert local_paths.resolve_presort_path(allowed, must_exist=True) == allowed.resolve()
    for outside in (tmp_path / "allowed-sibling", allowed / ".." / "outside"):
        with pytest.raises(ValueError, match="outside allowed roots"):
            local_paths.resolve_presort_path(outside)


def test_missing_allowed_path_rejected_before_read(monkeypatch, tmp_path):
    monkeypatch.setenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        local_paths.resolve_presort_path(tmp_path / "missing", must_exist=True)
