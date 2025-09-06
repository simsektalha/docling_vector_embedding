from pathlib import Path

from src.ingest.discover import discover_files


def test_discover_filters(tmp_path: Path):
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.docx").write_text("x")
    (tmp_path / "c.txt").write_text("x")

    files = discover_files(str(tmp_path), ["*.pdf", "*.docx"], ["c.*"], max_file_mb=1)
    names = sorted(Path(f.path).name for f in files)
    assert names == ["a.pdf", "b.docx"]


