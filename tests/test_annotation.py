from __future__ import annotations

import json
from pathlib import Path

from captcha_break.annotation import AnnotationWorkspace, Prelabel
from captcha_break.project_generator import PROJECT_ALPHABET

PNG = b"\x89PNG\r\n\x1a\nannotation-test"


def test_annotation_workspace_prepares_confirms_resumes_and_exports(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "captcha_0001.png").write_bytes(PNG)
    (source / "captcha_0001.json").write_text('{"label": null}', encoding="utf-8")
    workspace_path = tmp_path / "workspace"
    workspace = AnnotationWorkspace(workspace_path, alphabet=PROJECT_ALPHABET)

    calls = 0

    def predict(_image: bytes) -> Prelabel:
        nonlocal calls
        calls += 1
        return Prelabel("KJUU", "KJUV")

    assert workspace.prepare(source, predict) == 1
    assert workspace.state(0)["record"]["suggested_label"] == "KJUU"  # type: ignore[index]
    assert workspace.confirm(0, "kjuu") == 0
    assert workspace.progress()["confirmed"] == 1
    metadata = json.loads((workspace.images_dir / "captcha_0001.json").read_text(encoding="utf-8"))
    assert metadata["label"] == "KJUU"

    resumed = AnnotationWorkspace(workspace_path, alphabet=PROJECT_ALPHABET)
    assert resumed.prepare(source, predict) == 0
    assert calls == 1
    exported = resumed.export_confirmed()
    assert (exported / "KJUU_0001.png").read_bytes() == PNG
    assert (exported / "labels.csv").is_file()


def test_annotation_workspace_rejects_invalid_label(tmp_path: Path) -> None:
    workspace = AnnotationWorkspace(tmp_path, alphabet=PROJECT_ALPHABET)
    try:
        workspace.validate_label("O0O0")
    except ValueError as exc:
        assert "exactly 4" in str(exc)
    else:
        raise AssertionError("invalid characters must be rejected")
