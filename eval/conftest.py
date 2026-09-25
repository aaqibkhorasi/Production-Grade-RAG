"""Collects tracked-metric scores and renders them as a report.

Without this the scores exist only as pytest warning text buried in the CI log,
which has to be expanded and scrolled to read. On GitHub Actions the table is
written to the job summary so it renders on the run page; locally it prints to
the terminal after the run.
"""

import os
import statistics
from pathlib import Path

import pytest

METRIC_COLUMNS = ("Faithfulness", "Contextual Precision", "Contextual Relevancy")


class MetricRecorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, case_id: str, metric: str, score: float | None, threshold: float, error: str | None = None) -> None:
        self.rows.append(
            {"case_id": case_id, "metric": metric, "score": score, "threshold": threshold, "error": error}
        )


@pytest.fixture(scope="session")
def metric_recorder(request) -> MetricRecorder:
    recorder = MetricRecorder()
    request.session.metric_recorder = recorder
    return recorder


def _cell(row: dict | None) -> str:
    if row is None:
        return "–"
    if row["error"]:
        return "⚠️ judge error"
    score = row["score"]
    if score is None:
        return "–"
    # The threshold is advisory: these metrics are tracked, not gated.
    return f"{score:.2f}" if score >= row["threshold"] else f"{score:.2f} ⚠️"


def _render(rows: list[dict]) -> str:
    by_case: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["metric"]] = row

    lines = [
        "## RAG evaluation — tracked metrics",
        "",
        "Advisory signals, judged by an LLM. They do not gate the build; the hard",
        "gates are groundedness and citing the expected source document, which are",
        "ordinary pytest assertions and fail the run on their own.",
        "",
        "| Case | " + " | ".join(METRIC_COLUMNS) + " |",
        "| --- | " + " | ".join("---" for _ in METRIC_COLUMNS) + " |",
    ]
    for case_id in sorted(by_case):
        cells = [_cell(by_case[case_id].get(metric)) for metric in METRIC_COLUMNS]
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")

    lines += ["", "| Metric | Below threshold | Mean |", "| --- | --- | --- |"]
    for metric in METRIC_COLUMNS:
        scores = [r["score"] for r in rows if r["metric"] == metric and r["score"] is not None]
        if not scores:
            lines.append(f"| {metric} | – | – |")
            continue
        threshold = next(r["threshold"] for r in rows if r["metric"] == metric)
        below = sum(1 for s in scores if s < threshold)
        lines.append(f"| {metric} | {below} of {len(scores)} | {statistics.mean(scores):.3f} |")

    errors = [r for r in rows if r["error"]]
    if errors:
        lines += ["", f"{len(errors)} judge call(s) failed and were skipped:", ""]
        lines += [f"- `{r['case_id']}` / {r['metric']}: {r['error']}" for r in errors]
    return "\n".join(lines)


def pytest_sessionfinish(session, exitstatus) -> None:
    recorder = getattr(session, "metric_recorder", None)
    if recorder is None or not recorder.rows:
        return
    report = _render(recorder.rows)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(report + "\n")

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        reporter.write_line(report)
