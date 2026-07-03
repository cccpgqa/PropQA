#!/usr/bin/env python3
"""Recompute only the clean Open-NiCad baseline and merge with existing results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from no_leak_ground_truth import statement_ground_truth_from_files
from propagation_detector import GitHubClient
from localization_baselines import local_nicad
from rerun_rq5_filepair_v2 import local_nicad_files
from run_block_first_statement_experiment import metrics as statement_metrics
from run_file_pair_detection_experiment import (
    build_file_pair_records,
    metrics_for_records,
    restrict_state_to_source_file,
)
from run_graphqa_action_enhancement_experiment import file_metrics
from run_llm_in_loop_hard_cases import prepare_state
from run_no_leak_five_module_pipeline import (
    normalize_target_oracle_to_dataset_files,
    split_detector_state,
    split_label,
)
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv


def average(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    return {
        field: sum(float(row.get(field, 0.0)) for row in rows) / len(rows) if rows else 0.0
        for field in fields
    }


def aggregate(rows: list[dict[str, Any]], split: str, level: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["method"] == "Open-NiCad"
        and row["level"] == level
        and (split == "all" or row["split"] == split)
    ]
    fields = ["precision", "recall", "f1", "mrr"] if level == "file" else ["precision", "coverage", "f1", "mrr"]
    return {"split": split, "method": "Open-NiCad", "level": level, "pairs": len(selected), **average(selected, fields)}


def run_nicad_dataset(
    pairs: list[dict[str, Any]],
    label: str,
    client: GitHubClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    rows = []
    details = []
    snapshot_cache = {}
    for index, pair in enumerate(pairs, start=1):
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, oracle = split_detector_state(setup)
            split = split_label(pair)
            nicad_files, nicad_statements = local_nicad(
                state, runner, Path(args.git_root), 10, snapshot_cache
            )
            gt_files = set(oracle["target_changed_paths"])
            rows.append(
                {
                    "index": index,
                    "split": split,
                    "method": "Open-NiCad",
                    "level": "file",
                    **file_metrics([item["path"] for item in nicad_files], gt_files, 10),
                }
            )
            gt_statements = statement_ground_truth_from_files(oracle["target_files"])
            rows.append(
                {
                    "index": index,
                    "split": split,
                    "method": "Open-NiCad",
                    "level": "statement",
                    **statement_metrics(nicad_statements, gt_statements, 100, tolerance=2),
                }
            )
            details.append(
                {
                    "index": index,
                    "split": split,
                    "source_repo": state["propagator"].repo,
                    "target_repo": state["target"].repo,
                    "ground_truth_files": sorted(gt_files),
                    "open_nicad_predictions": [item["path"] for item in nicad_files],
                }
            )
            print(f"[{label} {index}/{len(pairs)}] Open-NiCad ok", flush=True)
        except Exception as exc:
            details.append({"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{label} {index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)
    splits = ["all", "simple", "hard"] if label == "url" else ["all"]
    summary = [aggregate(rows, split, level) for split in splits for level in ["file", "statement"]]
    return {"dataset": label, "pairs": len(pairs), "summary": summary, "rows": rows, "details": details}


def merge_localization_results(old_paths: list[Path], nicad_results: list[dict[str, Any]], out_path: Path) -> None:
    old_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in old_paths]
    old_by_dataset = {}
    for payload in old_payloads:
        for result in payload["results"]:
            old_by_dataset[result["dataset"]] = result
    nicad_by_dataset = {result["dataset"]: result for result in nicad_results}
    merged_results = []
    for dataset, old_result in old_by_dataset.items():
        nicad = nicad_by_dataset[dataset]
        rows = [row for row in old_result["rows"] if row["method"] != "Open-NiCad"] + nicad["rows"]
        summary = [row for row in old_result["summary"] if row["method"] != "Open-NiCad"] + nicad["summary"]
        merged_results.append({**old_result, "summary": summary, "rows": rows, "pure_open_nicad_details": nicad["details"]})
    configuration = {}
    for payload in old_payloads:
        configuration.update(payload.get("configuration", {}))
    payload = {**old_payloads[0], "configuration": {**configuration, "open_nicad_retrieval": "all target code files ranked only by token-clone similarity"}, "results": merged_results}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rq4_markdown(payload, out_path.with_suffix(".md"))


def write_rq4_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Localization Baseline Results",
        "",
        "Open-NiCad is recomputed as exhaustive all-target-code-file token-clone retrieval.",
        "No path prefilter, path score, candidate pruning, or target ground-truth hints are used.",
        "",
    ]
    order = {"Open-NiCad": 0, "code2vec": 1, "PropQA": 2}
    for result in payload["results"]:
        lines += [
            f"## {result['dataset']}",
            "",
            "| Split | Level | Method | Pairs | Precision | Recall/Coverage | F1 | MRR |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(result["summary"], key=lambda r: (r["split"], r["level"], order.get(r["method"], 9))):
            recall = row.get("recall", row.get("coverage", 0.0))
            lines.append(
                f"| {row['split']} | {row['level']} | {row['method']} | {row['pairs']} | "
                f"{row['precision']:.3f} | {recall:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_filepair_open_nicad(
    pairs: list[dict[str, Any]],
    client: GitHubClient,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    records = []
    predictions: dict[tuple[int, str, str], list[str]] = {}
    snapshot_cache = {}
    chunk_cache = {}
    errors = []
    for index, pair in enumerate(pairs, start=1):
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, _oracle = split_detector_state(setup)
            pair_records = build_file_pair_records(index, pair, setup)
            records.extend(pair_records)
            key = (state["target"].repo, state["target"].base_sha)
            if key not in snapshot_cache:
                from localization_baselines import clone_chunks, git_code_snapshot

                snapshot_cache[key] = git_code_snapshot(Path(args.git_root), state["target"].repo, state["target"].base_sha)
                chunk_cache[key] = {
                    path: clone_chunks(text[:24000])
                    for path, text in snapshot_cache[key].items()
                }
            for source_file in sorted({record["source_file"] for record in pair_records}):
                source_state = restrict_state_to_source_file(state, source_file)
                ranked = local_nicad_files(source_state, Path(args.git_root), 5, chunk_cache[key], snapshot_is_chunks=True)
                predictions[(index, source_file, "Open-NiCad")] = [item["path"] for item in ranked]
            print(f"[filepair {index}/{len(pairs)}] Open-NiCad ok", flush=True)
        except Exception as exc:
            errors.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[filepair {index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)
    import run_file_pair_detection_experiment as file_pair_module

    file_pair_module.args = SimpleNamespace(candidate_files=5)
    rows = metrics_for_records(records, predictions, ["Open-NiCad"], [1, 3, 5])
    return rows, records, errors


def merge_rq5(old_path: Path, open_rows: list[dict[str, Any]], records: list[dict[str, Any]], errors: list[dict[str, Any]], out_path: Path) -> None:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_file_rows = [row for row in old["file_pair_summary"] if row["method"] != "Open-NiCad"]
    payload = {
        **old,
        "open_nicad_retrieval": "all target code files ranked only by token-clone similarity",
        "file_pair_records": len(records),
        "pure_open_nicad_excluded": errors,
        "file_pair_summary": open_rows + old_file_rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rq5_markdown(payload, out_path.with_suffix(".md"))


def write_rq5_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# RQ5 Ablation and File-to-File Results",
        "",
        f"- Evaluated pairs: {payload['evaluated_pairs']}",
        f"- File-pair records: {payload['file_pair_records']}",
        "",
        "## Ablation",
        "",
        "| Split | Level | Variant | Pairs | Precision | Recall/Coverage | F1 | MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ablation_summary"]:
        recall = row.get("recall", row.get("coverage", 0.0))
        lines.append(
            f"| {row['split']} | {row['level']} | {row['method']} | {row['pairs']} | "
            f"{row['precision']:.3f} | {recall:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |"
        )
    lines += [
        "",
        "## File-to-File",
        "",
        "| Method | Metric | Value |",
        "|---|---|---:|",
    ]
    for row in payload["file_pair_summary"]:
        if row["metric"] != "MeanRank":
            lines.append(f"| {row['method']} | {row['metric']} | {row['value']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/detection_pairs_225.json")
    parser.add_argument("--old-url", default="outputs/rq4_rq5_repartitioned_v2.json")
    parser.add_argument("--old-rq5", default="outputs/rq5_filepair_repartitioned_v2.json")
    parser.add_argument("--out-rq4", default="outputs/rq4_pure_nicad.json")
    parser.add_argument("--out-rq5", default="outputs/rq5_filepair_repartitioned_v2_pure_nicad.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--git-root", default=".cache/git_repos")
    parser.add_argument("--mode", choices=["all", "rq4", "rq5"], default="all")
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    benchmark_pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.max_pairs:
        benchmark_pairs = benchmark_pairs[: args.max_pairs]
    if args.mode in {"all", "rq4"}:
        nicad_results = [
            run_nicad_dataset(benchmark_pairs, "detection_benchmark", client, args),
        ]
        merge_localization_results([Path(args.old_url)], nicad_results, Path(args.out_rq4))
        print(f"wrote {args.out_rq4}")
    if args.mode in {"all", "rq5"}:
        open_rows, records, errors = run_filepair_open_nicad(benchmark_pairs, client, args)
        merge_rq5(Path(args.old_rq5), open_rows, records, errors, Path(args.out_rq5))
        print(f"wrote {args.out_rq5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
