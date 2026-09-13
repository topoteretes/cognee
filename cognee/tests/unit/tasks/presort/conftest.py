import pytest


@pytest.fixture
def messy_folder(tmp_path):
    """A miniature messy Downloads folder."""
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake report body")
    (tmp_path / "report (1).pdf").write_bytes(b"%PDF-1.4 fake report body")  # exact dupe
    (tmp_path / "report_v2.pdf").write_bytes(b"%PDF-1.4 fake report body, revised")  # version
    (tmp_path / "notes.txt").write_text("Meet Ada at ada@example.com or +1 555 123 4567.")
    (tmp_path / "holiday.jpg").write_bytes(b"\xff\xd8\xff\xe0 fake jpeg bytes")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "partial.crdownload").write_bytes(b"partial download")
    (tmp_path / "empty.log").write_bytes(b"")

    invoices = tmp_path / "invoices"
    invoices.mkdir()
    (invoices / "invoice_march.pdf").write_bytes(b"%PDF invoice march")
    (invoices / "invoice_april.pdf").write_bytes(b"%PDF invoice april")

    project = tmp_path / "my_tool"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='my_tool'")
    (project / "main.py").write_text("print('hello')")

    hidden_dir = tmp_path / ".cache"
    hidden_dir.mkdir()
    (hidden_dir / "blob.bin").write_bytes(b"cached")

    return tmp_path
