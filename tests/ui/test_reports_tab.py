from pathlib import Path

import pytest
from gradio_app.app import list_report_files, read_report


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from market_research_team import config

    monkeypatch.setattr(config.settings, "reports_dir", tmp_path)
    return tmp_path


def test_list_report_files_returns_sorted_md_filenames(reports_dir: Path) -> None:
    (reports_dir / "b.md").write_text("# B")
    (reports_dir / "a.md").write_text("# A")
    (reports_dir / "notes.txt").write_text("ignore me")

    assert list_report_files() == ["a.md", "b.md"]


def test_list_report_files_empty_dir_returns_empty_list(reports_dir: Path) -> None:
    assert list_report_files() == []


def test_read_report_returns_file_contents(reports_dir: Path) -> None:
    (reports_dir / "a.md").write_text("# Report A\n\nBody.")

    assert read_report("a.md") == "# Report A\n\nBody."


def test_read_report_rejects_path_traversal(reports_dir: Path) -> None:
    with pytest.raises(ValueError):
        read_report("../secret.md")
