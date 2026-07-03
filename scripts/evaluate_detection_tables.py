#!/usr/bin/env python3
"""Build table-based evaluation for propagation localization experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TOP_KS = (3, 5, 8)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_files(result: dict[str, Any]) -> list[str]:
    pred = result.get("prediction") or {}
    if pred.get("target_files") is not None:
        return [item.get("path") for item in pred.get("target_files", []) if item.get("path")]
    return [item.get("path") for item in result.get("predicted_files", []) if item.get("path")]


def prediction_statements(result: dict[str, Any]) -> list[dict[str, Any]]:
    pred = result.get("prediction") or {}
    if pred.get("target_statements") is not None:
        return pred.get("target_statements", [])
    return result.get("predicted_statements", [])


def gt_files(result: dict[str, Any]) -> set[str]:
    gt = result.get("ground_truth") or {}
    if gt.get("target_changed_files") is not None:
        return set(gt.get("target_changed_files") or [])
    return set(result.get("target_changed_files") or [])


def gt_ranges(result: dict[str, Any]) -> list[tuple[str, int, int]]:
    gt = result.get("ground_truth") or {}
    ranges = gt.get("target_changed_old_line_ranges")
    out: list[tuple[str, int, int]] = []
    if ranges is None:
        ranges = result.get("ground_truth", {}).get("target_changed_old_line_ranges", {})
    if not ranges:
        ranges = result.get("metrics", {}).get("target_changed_old_line_ranges", {})
    for path, items in (ranges or {}).items():
        for item in items:
            out.append((path, int(item["start"]), int(item["end"])))
    return out


def line_hits_range(path: str, line: int, one_range: tuple[str, int, int], tolerance: int) -> bool:
    gt_path, start, end = one_range
    return path == gt_path and start - tolerance <= line <= end + tolerance


def split_results(results: list[dict[str, Any]], mode: str, hard_indices: set[int]) -> list[dict[str, Any]]:
    ok = [r for r in results if r.get("status") == "ok"]
    if mode == "all":
        return ok
    if mode == "simple":
        return [r for r in ok if int(r.get("index", -1)) not in hard_indices]
    if mode == "hard":
        return [r for r in ok if int(r.get("index", -1)) in hard_indices]
    raise ValueError(mode)


def file_metrics(results: list[dict[str, Any]], k: int) -> dict[str, float]:
    pair_hits = 0
    rr_sum = 0.0
    hit_total = 0
    pred_total = 0
    gt_total = 0
    macro_p = []
    macro_r = []
    for r in results:
        preds = prediction_files(r)[:k]
        gt = gt_files(r)
        if not gt:
            continue
        hits = [p for p in preds if p in gt]
        unique_hits = set(hits)
        if hits:
            pair_hits += 1
            first = next((i + 1 for i, p in enumerate(preds) if p in gt), None)
            if first:
                rr_sum += 1 / first
        hit_total += len(unique_hits)
        pred_total += len(preds)
        gt_total += len(gt)
        macro_p.append(len(unique_hits) / len(preds) if preds else 0.0)
        macro_r.append(len(unique_hits) / len(gt) if gt else 0.0)
    precision = hit_total / pred_total if pred_total else 0.0
    recall = hit_total / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pairs": len(results),
        "hit_rate": pair_hits / len(results) if results else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": rr_sum / len(results) if results else 0.0,
        "macro_precision": sum(macro_p) / len(macro_p) if macro_p else 0.0,
        "macro_recall": sum(macro_r) / len(macro_r) if macro_r else 0.0,
    }


def statement_metrics(results: list[dict[str, Any]], k: int, tolerance: int) -> dict[str, float]:
    pair_hits = 0
    rr_sum = 0.0
    hit_pred_total = 0
    pred_total = 0
    covered_range_total = 0
    gt_range_total = 0
    macro_p = []
    macro_r = []
    for r in results:
        preds = prediction_statements(r)[:k]
        ranges = gt_ranges(r)
        if not ranges:
            continue
        pred_hits = []
        covered = set()
        for idx, item in enumerate(preds):
            path = item.get("path")
            try:
                line = int(item.get("line"))
            except (TypeError, ValueError):
                continue
            matched = [rid for rid, one in enumerate(ranges) if line_hits_range(path, line, one, tolerance)]
            if matched:
                pred_hits.append(idx)
                covered.update(matched)
        if pred_hits:
            pair_hits += 1
            rr_sum += 1 / (pred_hits[0] + 1)
        hit_pred_total += len(pred_hits)
        pred_total += len(preds)
        covered_range_total += len(covered)
        gt_range_total += len(ranges)
        macro_p.append(len(pred_hits) / len(preds) if preds else 0.0)
        macro_r.append(len(covered) / len(ranges) if ranges else 0.0)
    precision = hit_pred_total / pred_total if pred_total else 0.0
    recall = covered_range_total / gt_range_total if gt_range_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pairs": len(results),
        "hit_rate": pair_hits / len(results) if results else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": rr_sum / len(results) if results else 0.0,
        "macro_precision": sum(macro_p) / len(macro_p) if macro_p else 0.0,
        "macro_recall": sum(macro_r) / len(macro_r) if macro_r else 0.0,
    }


def hard_indices_from(path: Path) -> set[int]:
    if not path.exists():
        return set()
    data = load(path)
    return {int(item["index"]) for item in data.get("hard_cases", []) if item.get("index") is not None}


def format_float(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    headers = ["Dataset", "Method", "Level", "K", "Pairs", "Hit@K", "Precision", "Recall", "F1", "MRR"]
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    str(row["level"]),
                    str(row["k"]),
                    str(row["pairs"]),
                    format_float(row["hit_rate"]),
                    format_float(row["precision"]),
                    format_float(row["recall"]),
                    format_float(row["f1"]),
                    format_float(row["mrr"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def collect_rows(
    experiment_name: str,
    method_name: str,
    results: list[dict[str, Any]],
    hard_indices: set[int],
    tolerance: int,
    datasets: tuple[str, ...] = ("all", "simple", "hard"),
) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        subset = split_results(results, dataset, hard_indices)
        if not subset:
            continue
        for level, metric_fn in (("file", file_metrics), ("statement", statement_metrics)):
            for k in TOP_KS:
                metrics = metric_fn(subset, k, tolerance) if level == "statement" else metric_fn(subset, k)
                rows.append(
                    {
                        "experiment": experiment_name,
                        "dataset": dataset,
                        "method": method_name,
                        "level": level,
                        "k": k,
                        **metrics,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ast", default="outputs/qa_mcts_ast_detection_subset_top8.json")
    parser.add_argument("--baseline", default="outputs/propagation_baseline_detection_subset_top8.json")
    parser.add_argument("--llm", default="outputs/qa_mcts_llm_forced_hard6_top20_max4000.json")
    parser.add_argument("--llm-loop", default="outputs/qa_mcts_llm_in_loop_hard15.json")
    parser.add_argument("--hard-source", default="outputs/qa_mcts_ast_detection_subset_full_merged.json")
    parser.add_argument("--output-json", default="outputs/detection_evaluation_tables.json")
    parser.add_argument("--output-md", default="outputs/detection_evaluation_tables.md")
    parser.add_argument("--output-csv", default="outputs/detection_evaluation_tables.csv")
    parser.add_argument("--tolerance", type=int, default=5)
    args = parser.parse_args()

    hard_indices = hard_indices_from(Path(args.hard_source))
    all_rows: list[dict[str, Any]] = []

    ast_path = Path(args.ast)
    if ast_path.exists():
        ast_results = load(ast_path).get("results", [])
        all_rows.extend(collect_rows("main", "PropQA-AST", ast_results, hard_indices, args.tolerance))

    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        baseline_results = load(baseline_path).get("results", [])
        all_rows.extend(collect_rows("ablation", "Path+Line baseline", baseline_results, hard_indices, args.tolerance))

    llm_path = Path(args.llm)
    if llm_path.exists():
        llm_results = load(llm_path).get("results", [])
        llm_indices = {int(r["index"]) for r in llm_results if r.get("status") == "ok"}
        if ast_path.exists():
            ast_same = [r for r in ast_results if int(r.get("index", -1)) in llm_indices]
            all_rows.extend(collect_rows("llm_ablation", "PropQA-AST same hard6", ast_same, llm_indices, args.tolerance, datasets=("hard",)))
        all_rows.extend(collect_rows("llm_ablation", "PropQA-AST+LLM hard6", llm_results, llm_indices, args.tolerance, datasets=("hard",)))

    llm_loop_path = Path(args.llm_loop)
    if llm_loop_path.exists():
        loop_results = load(llm_loop_path).get("results", [])
        loop_indices = {int(r["index"]) for r in loop_results if r.get("status") == "ok"}
        if ast_path.exists():
            ast_same_loop = [r for r in ast_results if int(r.get("index", -1)) in loop_indices]
            all_rows.extend(collect_rows("llm_loop", "PropQA-AST same hard15", ast_same_loop, loop_indices, args.tolerance, datasets=("hard",)))
        all_rows.extend(collect_rows("llm_loop", "LLM-in-loop hard15", loop_results, loop_indices, args.tolerance, datasets=("hard",)))

    Path(args.output_json).write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        fields = ["experiment", "dataset", "method", "level", "k", "pairs", "hit_rate", "precision", "recall", "f1", "mrr", "macro_precision", "macro_recall"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    main_rows = [r for r in all_rows if r["experiment"] == "main"]
    ablation_rows = [r for r in all_rows if r["experiment"] == "ablation"]
    llm_rows = [r for r in all_rows if r["experiment"] == "llm_ablation"]
    llm_loop_rows = [r for r in all_rows if r["experiment"] == "llm_loop"]
    md_parts = [
        "# Detection Evaluation Tables",
        "",
        "Metrics are micro-averaged unless otherwise noted. File recall is the fraction of ground-truth changed files retrieved. Statement recall is the fraction of target changed-line ranges covered within +/-5 lines.",
        "",
        markdown_table("Main PropQA-AST Results", main_rows),
        "",
        markdown_table("Ablation: Path+Line Baseline", ablation_rows),
        "",
        markdown_table("Ablation: LLM Reranking on Hard6", llm_rows),
        "",
        markdown_table("Ablation: LLM-in-the-loop on Hard15", llm_loop_rows),
        "",
    ]
    Path(args.output_md).write_text("\n".join(md_parts), encoding="utf-8")
    print(f"rows {len(all_rows)}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
