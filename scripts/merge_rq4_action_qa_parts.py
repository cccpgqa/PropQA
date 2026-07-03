"""Merge disjoint RQ4 action-QA experiment parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def weighted(rows: list[tuple[dict[str, Any], int]], fields: tuple[str, ...]) -> dict[str, Any]:
    total = sum(weight for _row, weight in rows)
    return {
        "pairs": total,
        **{
            field: sum(float(row[field]) * weight for row, weight in rows) / total if total else 0.0
            for field in fields
        },
    }


def merge_file(parts: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_rows = [row for row in parts[0]["rows"] if row["method"] != "PropQA"]
    propqa_by_metric: dict[str, list[tuple[float, int]]] = {}
    for part in parts:
        for row in part["rows"]:
            if row["method"] == "PropQA":
                propqa_by_metric.setdefault(row["metric"], []).append(
                    (float(row["value"]), int(row["file_pairs"]))
                )
    propqa_rows = []
    for metric, values in propqa_by_metric.items():
        count = sum(weight for _value, weight in values)
        propqa_rows.append(
            {
                "method": "PropQA",
                "metric": metric,
                "value": sum(value * weight for value, weight in values) / count,
                "file_pairs": count,
            }
        )
    return {
        "dataset": "disjoint RQ4 chunks",
        "evaluated_pairs": sum(int(part["evaluated_pairs"]) for part in parts),
        "file_pair_records": sum(int(part["file_pair_records"]) for part in parts),
        "llm": parts[0]["llm"],
        "rows": baseline_rows + propqa_rows,
        "llm_failures": sum(int(part["llm_failures"]) for part in parts),
        "details": [detail for part in parts for detail in part["details"]],
        "errors": [error for part in parts for error in part["errors"]],
    }


def merge_statement(parts: list[dict[str, Any]]) -> dict[str, Any]:
    method_names = sorted({method for part in parts for method in part["summary"]})
    summary = {}
    for method in method_names:
        rows = [
            (part["summary"][method], int(part["summary"][method]["pairs"]))
            for part in parts
            if method in part["summary"]
        ]
        summary[method] = weighted(rows, ("precision", "coverage", "f1", "mrr"))
    return {
        "dataset": "disjoint RQ4 chunks",
        "task": parts[0]["task"],
        "llm": parts[0]["llm"],
        "evaluated_pairs": sum(int(part["evaluated_pairs"]) for part in parts),
        "ground_truth_counts": {
            key: sum(int(part["ground_truth_counts"][key]) for part in parts)
            for key in ("all", "edited", "deleted")
        },
        "summary": summary,
        "details": [detail for part in parts for detail in part["details"]],
        "errors": [error for part in parts for error in part["errors"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("file", "statement"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("parts", nargs="+", type=Path)
    args = parser.parse_args()
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in args.parts]
    payload = merge_file(parts) if args.level == "file" else merge_statement(parts)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
