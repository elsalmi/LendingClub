#!/usr/bin/env python3
"""Render the short LendingClub fairness report from the committed model card.

The notebook export in ``Fairness.md`` remains the underlying evidence, while
``MODEL_CARD.md`` is the structured snapshot this script reads to regenerate
``reports/FAIRNESS_REPORT.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD = ROOT / "MODEL_CARD.md"
DEFAULT_OUTPUT = ROOT / "reports" / "FAIRNESS_REPORT.md"


def extract_table_after_marker(text: str, marker: str) -> Tuple[List[str], List[List[str]]]:
    try:
        start = text.index(marker)
    except ValueError as exc:
        raise SystemExit(f"Could not find marker: {marker!r}") from exc

    tail = text[start + len(marker) :].splitlines()
    table_lines: List[str] = []
    seen_row = False
    for line in tail:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            seen_row = True
            continue
        if seen_row and not stripped:
            break
        if seen_row and not stripped.startswith("|"):
            break

    if len(table_lines) < 3:
        raise SystemExit(f"Did not find a full markdown table after {marker!r}")

    headers = [col.strip() for col in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        rows.append([col.strip() for col in line.strip("|").split("|")])
    return headers, rows


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(row: Sequence[str]) -> str:
        cells = [f" {cell.ljust(widths[idx])} " for idx, cell in enumerate(row)]
        return "|" + "|".join(cells) + "|"

    header_line = fmt_row(headers)
    separator = "|" + "|".join(["-" * (w + 2) for w in widths]) + "|"
    body = [fmt_row(row) for row in rows]
    return "\n".join([header_line, separator, *body])


def build_report() -> str:
    text = MODEL_CARD.read_text()
    metrics_headers, metrics_rows = extract_table_after_marker(
        text, "## Metrics reported from existing artifacts"
    )
    reweigh_headers, reweigh_rows = extract_table_after_marker(
        text, "Reported mean-outcome differences:"
    )

    metrics_table = render_table(metrics_headers, metrics_rows)
    reweigh_table = render_table(reweigh_headers, reweigh_rows)

    return f"""# Fairness Report - LendingClub

This short report is generated from the committed evidence in
[MODEL_CARD.md](../MODEL_CARD.md) and traces back to the notebook export in
[Fairness.md](../Fairness.md).

## Executive summary

This project predicts whether an approved LendingClub loan will be `Fully Paid`
or `Charged Off`, then checks whether outcomes differ across a ZIP3-derived
demographic proxy. Reweighing reduced the reported training mean outcome
difference from `-0.017565` to `0.000000`, and reduced the reported test mean
outcome difference from `-0.019698` to `-0.002036`.

## Prediction task

- Positive label: `Fully Paid`.
- Negative label: `Charged Off`.
- Protected-proxy attribute: `underprivileged`, derived from ZIP3-linked Census
  racial composition.
- Fairness toolkit: AIF360.

## Metric snapshot

{metrics_table}

## Reweighing snapshot

{reweigh_table}

## Key risks

- ZIP3 proxy groups can expose aggregate disparity, but they are not individual
  protected-class labels.
- Reweighing reduces one measured disparity but does not guarantee fairness
  under every threshold, metric, or deployment setting.
- The raw data and runtime are not yet packaged for one-command reproduction.

## Regeneration

Run `python scripts/build_fairness_report.py --write` to rewrite this file from
the committed model-card snapshot.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Write the report to {DEFAULT_OUTPUT} instead of stdout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path when --write is used.",
    )
    args = parser.parse_args()

    report = build_report()
    if args.write:
        args.output.write_text(report)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
