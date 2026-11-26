"""Exercise the public CLI against the bundled offline tool traces."""

import json
from pathlib import Path

import pytest
from rich.text import Text

from openagent_eval.cli.main import app

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "tool-call-correctness"


@pytest.mark.parametrize(("threshold", "exit_code"), [("0.5", 0), ("1", 1)])
def test_recorded_tool_call_gate(runner, monkeypatch, threshold, exit_code):
    monkeypatch.chdir(EXAMPLE.parents[1])
    result = runner.invoke(
        app,
        [
            "test",
            str(EXAMPLE / "config.yaml"),
            "-t",
            f"tool_call_correctness:gte:{threshold}",
            "--json",
        ],
    )
    assert result.exit_code == exit_code, result.output
    assert '"actual": 0.5' in Text.from_ansi(result.output).plain


def test_diagnose_recorded_tool_calls_from_saved_report(runner, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        (EXAMPLE / "config.yaml")
        .read_text()
        .replace(
            "examples/tool-call-correctness/traces.jsonl", str(EXAMPLE / "traces.jsonl")
        ),
        encoding="utf-8",
    )

    evaluation = runner.invoke(app, ["run", str(config)])
    assert evaluation.exit_code == 0, evaluation.output
    (path,) = (tmp_path / "reports").glob("*.json")
    result = runner.invoke(app, ["diagnose", str(path), "--output", "json"])

    assert result.exit_code == 0, result.output
    report = json.loads(Text.from_ansi(result.output).plain)
    assert report["blame_summary"] == {"tool_call": 2}
    assert report["overall_health"] == pytest.approx(1 / 3)
    assert [(f["mode"], f["step"]) for f in report["failures"]] == [
        ("wrong_tool_arguments", 1),
        ("wrong_tool_call", 1),
    ]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "matter_id" not in json.dumps(saved)
