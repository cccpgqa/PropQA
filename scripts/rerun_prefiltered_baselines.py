#!/usr/bin/env python3
"""Evaluate path-prefiltered code2vec and Open-NiCad baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from no_leak_ground_truth import statement_ground_truth_from_files
from propagation_detector import GitHubClient
from localization_baselines import (
    ExhaustiveCode2Vec,
    clone_chunks,
    clone_similarity_against_target_chunks,
    clone_similarity_from_chunks,
    git_file,
    is_code_file,
)
from run_agent_strategy_matrix import map_candidates
from run_block_first_statement_experiment import metrics as statement_metrics
from run_file_pair_detection_experiment import (
    build_file_pair_records,
    metrics_for_records,
    restrict_state_to_source_file,
)
from run_graphqa_action_enhancement_experiment import file_metrics
from run_llm_in_loop_hard_cases import prepare_state
from run_new_hard_baseline_ablation import path_prefilter, source_patch_text
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


def aggregate(rows: list[dict[str, Any]], split: str, method: str, level: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["method"] == method
        and row["level"] == level
        and (split == "all" or row["split"] == split)
    ]
    fields = ["precision", "recall", "f1", "mrr"] if level == "file" else ["precision", "coverage", "f1", "mrr"]
    return {"split": split, "method": method, "level": level, "pairs": len(selected), **average(selected, fields)}


def open_nicad_prefilter(
    state: dict[str, Any],
    git_root: Path,
    top: int,
    prefilter_limit: int,
) -> list[dict[str, Any]]:
    source_chunks = clone_chunks(source_patch_text(state), max_chunks=80)
    ranked = []
    for item in path_prefilter(state, prefilter_limit):
        path = item["path"]
        if not is_code_file(path):
            continue
        text = git_file(git_root, state["target"].repo, state["target"].base_sha, path)
        score = clone_similarity_from_chunks(source_chunks, text) if text else 0.0
        ranked.append(
            {
                "path": path,
                "score": score,
                "matched_source_path": item.get("matched_source_path"),
                "evidence": {"path_prefilter_rank_score": item.get("score"), "reranker": "Open-NiCad"},
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), item["path"]))
    return ranked[:top]


def code2vec_prefilter(
    state: dict[str, Any],
    code2vec: ExhaustiveCode2Vec,
    top: int,
    prefilter_limit: int,
) -> list[dict[str, Any]]:
    source_vectors = []
    source_paths = []
    for path in state.get("source_changed_paths", []):
        vector = code2vec.blob_vector(state["propagator"].repo, state["propagator"].merged_sha, path)
        if vector is not None:
            source_vectors.append(vector)
            source_paths.append(path)
    if not source_vectors:
        return []
    source_matrix = np.stack(source_vectors)
    ranked = []
    for item in path_prefilter(state, prefilter_limit):
        path = item["path"]
        if not path.endswith(".go"):
            continue
        vector = code2vec.blob_vector(state["target"].repo, state["target"].base_sha, path)
        if vector is None:
            continue
        similarities = source_matrix @ vector
        best = int(np.argmax(similarities))
        ranked.append(
            {
                "path": path,
                "score": float(similarities[best]),
                "matched_source_path": source_paths[best],
                "evidence": {"path_prefilter_rank_score": item.get("score"), "reranker": "code2vec"},
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), item["path"]))
    return ranked[:top]


def open_nicad_prefilter_filequery(
    state: dict[str, Any],
    git_root: Path,
    top: int,
    prefilter_limit: int,
) -> list[dict[str, Any]]:
    source_parts = [
        git_file(git_root, state["propagator"].repo, state["propagator"].merged_sha, path)
        for path in state.get("source_changed_paths", [])
        if is_code_file(path)
    ]
    source_text = "\n".join(part for part in source_parts if part) or source_patch_text(state)
    source_chunks = clone_chunks(source_text, max_chunks=80)
    ranked = []
    for item in path_prefilter(state, prefilter_limit):
        path = item["path"]
        if not is_code_file(path):
            continue
        text = git_file(git_root, state["target"].repo, state["target"].base_sha, path)
        chunks = clone_chunks(text[:24000]) if text else []
        if not chunks:
            continue
        score = clone_similarity_against_target_chunks(source_chunks, chunks)
        ranked.append({"path": path, "score": score, "matched_source_path": item.get("matched_source_path")})
    ranked.sort(key=lambda item: (-float(item["score"]), item["path"]))
    return ranked[:top]


def run_patch_and_statement(
    pairs: list[dict[str, Any]],
    label: str,
    client: GitHubClient,
    code2vec: ExhaustiveCode2Vec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    rows = []
    details = []
    git_root = Path(args.git_root)
    for index, pair in enumerate(pairs, start=1):
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, oracle = split_detector_state(setup)
            split = split_label(pair)
            gt_files = set(oracle["target_changed_paths"])
            methods = {
                "Open-NiCad+path_prefilter": open_nicad_prefilter(state, git_root, 10, args.prefilter_files),
                "code2vec+path_prefilter": code2vec_prefilter(state, code2vec, 10, args.prefilter_files),
            }
            for method, predictions in methods.items():
                rows.append(
                    {
                        "index": index,
                        "split": split,
                        "method": method,
                        "level": "file",
                        **file_metrics([item["path"] for item in predictions], gt_files, 10),
                    }
                )
            gt_statements = statement_ground_truth_from_files(oracle["target_files"])
            for method, predictions in methods.items():
                statements = map_candidates(runner, state, predictions, 200)[:100]
                rows.append(
                    {
                        "index": index,
                        "split": split,
                        "method": method,
                        "level": "statement",
                        **statement_metrics(statements, gt_statements, 100, tolerance=2),
                    }
                )
            details.append({"index": index, "status": "ok"})
            print(f"[{label} {index}/{len(pairs)}] ok", flush=True)
        except Exception as exc:
            details.append({"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{label} {index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)
    splits = ["all", "simple", "hard"] if label == "url" else ["all"]
    methods = ["Open-NiCad+path_prefilter", "code2vec+path_prefilter"]
    summary = [
        aggregate(rows, split, method, level)
        for split in splits
        for level in ["file", "statement"]
        for method in methods
    ]
    return {"dataset": label, "pairs": len(pairs), "summary": summary, "rows": rows, "details": details}


def run_filepair(
    pairs: list[dict[str, Any]],
    client: GitHubClient,
    code2vec: ExhaustiveCode2Vec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    git_root = Path(args.git_root)
    records = []
    predictions: dict[tuple[int, str, str], list[str]] = {}
    errors = []
    for index, pair in enumerate(pairs, start=1):
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, _oracle = split_detector_state(setup)
            pair_records = build_file_pair_records(index, pair, setup)
            records.extend(pair_records)
            for source_file in sorted({record["source_file"] for record in pair_records}):
                source_state = restrict_state_to_source_file(state, source_file)
                c2v = code2vec_prefilter(source_state, code2vec, 5, args.prefilter_files)
                nicad = open_nicad_prefilter_filequery(source_state, git_root, 5, args.prefilter_files)
                predictions[(index, source_file, "code2vec+path_prefilter")] = [item["path"] for item in c2v]
                predictions[(index, source_file, "Open-NiCad+path_prefilter")] = [item["path"] for item in nicad]
            print(f"[filepair {index}/{len(pairs)}] ok", flush=True)
        except Exception as exc:
            errors.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[filepair {index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)
    import run_file_pair_detection_experiment as file_pair_module

    file_pair_module.args = SimpleNamespace(candidate_files=5)
    summary = metrics_for_records(
        records,
        predictions,
        ["Open-NiCad+path_prefilter", "code2vec+path_prefilter"],
        [1, 3, 5],
    )
    return {"file_pair_records": len(records), "summary": summary, "errors": errors}


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Path-Prefiltered Baseline Results",
        "",
        "Path prefiltering is used only to define the candidate pool; final ranking uses clone similarity or code2vec cosine similarity.",
        "",
    ]
    for result in payload["patch_statement_results"]:
        lines += [
            f"## {result['dataset']}",
            "",
            "| Split | Level | Method | Pairs | Precision | Recall/Coverage | F1 | MRR |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in result["summary"]:
            recall = row.get("recall", row.get("coverage", 0.0))
            lines.append(
                f"| {row['split']} | {row['level']} | {row['method']} | {row['pairs']} | "
                f"{row['precision']:.3f} | {recall:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |"
            )
        lines.append("")
    if payload.get("filepair_result"):
        lines += [
            "## file-to-file",
            "",
            "| Method | Metric | Value |",
            "|---|---|---:|",
        ]
        for row in payload["filepair_result"]["summary"]:
            if row["metric"] != "MeanRank":
                lines.append(f"| {row['method']} | {row['metric']} | {row['value']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/detection_pairs_225.json")
    parser.add_argument("--checkpoint", default=".cache/go_code2vec/geth_go_code2vec.pt")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--git-root", default=".cache/git_repos")
    parser.add_argument("--vector-cache", default=".cache/go_code2vec/blob_vectors_ctx12_term8")
    parser.add_argument("--output", default="outputs/path_prefiltered_baselines.json")
    parser.add_argument("--prefilter-files", type=int, default=80)
    parser.add_argument("--mode", choices=["all", "rq4", "rq5"], default="all")
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    code2vec = ExhaustiveCode2Vec(Path(args.checkpoint), Path(args.git_root), Path(args.vector_cache))
    benchmark_pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.max_pairs:
        benchmark_pairs = benchmark_pairs[: args.max_pairs]
    payload: dict[str, Any] = {
        "configuration": {
            "prefilter_files": args.prefilter_files,
            "ranking": "path_prefilter candidate pool, then method-only reranking",
        },
        "patch_statement_results": [],
        "filepair_result": None,
    }
    if args.mode in {"all", "rq4"}:
        payload["patch_statement_results"] = [
            run_patch_and_statement(benchmark_pairs, "detection_benchmark", client, code2vec, args),
        ]
    if args.mode in {"all", "rq5"}:
        payload["filepair_result"] = run_filepair(benchmark_pairs, client, code2vec, args)
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
