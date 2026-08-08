"""Renders data quality reports (Markdown + HTML) from a ValidationResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditguard.validation.engine import ValidationResult

TOP_N_EXAMPLES = 20
Tables = dict[str, pd.DataFrame]


def _numeric_summary(tables: Tables) -> pd.DataFrame:
    """Per (table, column) summary stats for every numeric column across `tables`."""
    rows = []
    for table, df in tables.items():
        numeric_df = df.select_dtypes(include=[np.number])
        for column in numeric_df.columns:
            desc = numeric_df[column].describe()
            rows.append(
                {
                    "table": table,
                    "column": column,
                    "count": int(desc.get("count", 0)),
                    "mean": desc.get("mean"),
                    "std": desc.get("std"),
                    "min": desc.get("min"),
                    "median": numeric_df[column].median(),
                    "max": desc.get("max"),
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _rule_violation_table(result: ValidationResult) -> pd.DataFrame:
    """Rule -> table -> severity -> count -> % of that table's rows."""
    if result.rule_counts.empty:
        return result.rule_counts.assign(pct_of_rows=pd.Series(dtype=float))
    counts = result.rule_counts.copy()
    counts["pct_of_rows"] = counts.apply(
        lambda r: (
            100.0 * r["count"] / result.row_counts.get(r["table"], 1)
            if result.row_counts.get(r["table"])
            else 0.0
        ),
        axis=1,
    )
    return counts


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "_none_\n"
    columns = columns or list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(
    dataset_version: str,
    result: ValidationResult,
    before: Tables,
    after: Tables | None = None,
) -> str:
    """Render the full data quality report as Markdown."""
    lines = [f"# Data quality report: {dataset_version}", ""]

    lines.append("## Row counts")
    lines.append("")
    row_rows = []
    for table, before_count in result.row_counts.items():
        quarantined = len(result.quarantined_keys.get(table, set()))
        after_count = (
            len(after[table]) if after is not None and table in after else None
        )
        row_rows.append(
            {
                "table": table,
                "rows_in": before_count,
                "error_flagged": quarantined,
                "rows_out": (
                    after_count
                    if after_count is not None
                    else before_count - quarantined
                ),
            }
        )
    lines.append(_markdown_table(pd.DataFrame(row_rows)))

    lines.append("## Rule violations")
    lines.append("")
    rule_table = _rule_violation_table(result)
    lines.append(
        _markdown_table(
            rule_table,
            columns=["rule_name", "table", "severity", "count", "pct_of_rows"],
        )
    )

    lines.append(f"## Top {TOP_N_EXAMPLES} example violations")
    lines.append("")
    examples = result.violations.sort_values("severity").head(TOP_N_EXAMPLES)
    lines.append(
        _markdown_table(
            examples, columns=["table", "record_key", "rule_name", "severity", "detail"]
        )
    )

    lines.append("## Numeric column statistics (before)")
    lines.append("")
    before_stats = _numeric_summary(before)
    lines.append(_markdown_table(before_stats))

    if after is not None:
        lines.append("## Numeric column statistics (after cleaning)")
        lines.append("")
        after_stats = _numeric_summary(after)
        lines.append(_markdown_table(after_stats))

    status = "PASS" if result.passed else "FAIL (ERROR violations present)"
    lines.append(f"**Overall result:** {status}")
    lines.append("")
    return "\n".join(lines)


def _html_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "<p><em>none</em></p>"
    columns = columns or list(df.columns)
    head = "".join(f"<th>{c}</th>" for c in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(
    dataset_version: str,
    result: ValidationResult,
    before: Tables,
    after: Tables | None = None,
) -> str:
    """Render the full data quality report as a standalone HTML page."""
    row_rows = []
    for table, before_count in result.row_counts.items():
        quarantined = len(result.quarantined_keys.get(table, set()))
        after_count = (
            len(after[table]) if after is not None and table in after else None
        )
        row_rows.append(
            {
                "table": table,
                "rows_in": before_count,
                "error_flagged": quarantined,
                "rows_out": (
                    after_count
                    if after_count is not None
                    else before_count - quarantined
                ),
            }
        )

    rule_table = _rule_violation_table(result)
    examples = result.violations.sort_values("severity").head(TOP_N_EXAMPLES)
    before_stats = _numeric_summary(before)
    status = "PASS" if result.passed else "FAIL (ERROR violations present)"

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Data quality report: {dataset_version}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a;}",
        "table{border-collapse:collapse;margin-bottom:1.5rem;width:100%;}",
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right;"
        "font-size:0.85rem;}",
        "th:first-child,td:first-child{text-align:left;}",
        "th{background:#f0f0f0;}",
        "h2{margin-top:2rem;}",
        "</style>",
        f"<h1>Data quality report: {dataset_version}</h1>",
        f"<p><strong>Overall result:</strong> {status}</p>",
        "<h2>Row counts</h2>",
        _html_table(pd.DataFrame(row_rows)),
        "<h2>Rule violations</h2>",
        _html_table(
            rule_table,
            columns=["rule_name", "table", "severity", "count", "pct_of_rows"],
        ),
        f"<h2>Top {TOP_N_EXAMPLES} example violations</h2>",
        _html_table(
            examples, columns=["table", "record_key", "rule_name", "severity", "detail"]
        ),
        "<h2>Numeric column statistics (before)</h2>",
        _html_table(before_stats),
    ]
    if after is not None:
        after_stats = _numeric_summary(after)
        parts.append("<h2>Numeric column statistics (after cleaning)</h2>")
        parts.append(_html_table(after_stats))
    return "\n".join(parts)


def write_report(
    dataset_version: str,
    result: ValidationResult,
    before: Tables,
    after: Tables | None,
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Write the Markdown and HTML reports to reports_dir/<version>.{md,html}."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{dataset_version}.md"
    html_path = reports_dir / f"{dataset_version}.html"
    md_path.write_text(
        render_markdown(dataset_version, result, before, after), encoding="utf-8"
    )
    html_path.write_text(
        render_html(dataset_version, result, before, after), encoding="utf-8"
    )
    return md_path, html_path
