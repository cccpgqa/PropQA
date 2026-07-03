#!/usr/bin/env python3
"""Re-run PropQA ablations and file-to-file localization on dataset V2."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from no_leak_ground_truth import statement_ground_truth_from_files
from propagation_detector import GitHubClient, token_similarity
from localization_baselines import (
    ExhaustiveCode2Vec,
    clone_chunks,
    clone_similarity_against_target_chunks,
    clone_similarity_from_chunks,
    git_code_snapshot,
    git_file,
    is_code_file,
)
from run_block_first_statement_experiment import metrics as statement_metrics
from run_file_pair_detection_experiment import (
    build_file_pair_records,
    graphqa_action_files,
    metrics_for_records,
    restrict_state_to_source_file,
)
from run_graphqa_action_enhancement_experiment import (
    direct_intent_ast_blocks,
    enhanced_file_rank,
    expand_direct_blocks,
    file_metrics,
)
from run_llm_in_loop_hard_cases import prepare_state
from run_new_hard_baseline_ablation import (
    action_graphqa,
    source_patch_text,
)
from run_no_leak_five_module_pipeline import (
    normalize_target_oracle_to_dataset_files,
    split_detector_state,
    split_label,
)
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv, meaningful_statement


def average(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    return {
        field: sum(float(row.get(field, 0.0)) for row in rows) / len(rows) if rows else 0.0
        for field in fields
    }


def lexical_statement_baseline(
    state: dict[str, Any],
    target_paths: list[str],
    git_root: Path,
    top: int,
) -> list[dict[str, Any]]:
    source_texts = [
        statement.text
        for statement in state.get("source_statements", [])
        if meaningful_statement(statement.text)
    ]
    ranked = []
    for path in target_paths:
        text = git_file(git_root, state["target"].repo, state["target"].base_sha, path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not meaningful_statement(stripped):
                continue
            score = max(
                (token_similarity(source, stripped) for source in source_texts),
                default=0.0,
            )
            ranked.append({"path": path, "line": line_no, "score": score})
    ranked.sort(key=lambda item: (-item["score"], item["path"], item["line"]))
    return ranked[:top]


def local_nicad_files(
    state: dict[str, Any],
    git_root: Path,
    top: int,
    code_snapshot: dict[str, str] | dict[str, list[set[str]]],
    snapshot_is_chunks: bool = False,
) -> list[dict[str, Any]]:
    source_parts = [
        git_file(git_root, state["propagator"].repo, state["propagator"].merged_sha, path)
        for path in state.get("source_changed_paths", [])
        if is_code_file(path)
    ]
    source_text = "\n".join(part for part in source_parts if part) or source_patch_text(state)
    source_chunks = clone_chunks(source_text, max_chunks=80)
    ranked = []
    for path, text_or_chunks in code_snapshot.items():
        if snapshot_is_chunks:
            clone = clone_similarity_against_target_chunks(source_chunks, text_or_chunks) if text_or_chunks else 0.0
        else:
            clone = clone_similarity_from_chunks(source_chunks, text_or_chunks) if text_or_chunks else 0.0
        ranked.append(
            {
                "path": path,
                "score": clone,
                "matched_source_path": None,
                "evidence": {"retrieval_scope": "all_target_code_files", "path_score": None},
                "baseline": "Open-NiCad",
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), item["path"]))
    return ranked[:top]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/URL_Results_detection_url_celo_v2.json")
    parser.add_argument("--checkpoint", default=".cache/go_code2vec/geth_go_code2vec.pt")
    parser.add_argument("--output", default="outputs/rq5_filepair_repartitioned_v2.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--git-root", default=".cache/git_repos")
    parser.add_argument("--vector-cache", default=".cache/go_code2vec/blob_vectors_ctx12_term8")
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    args.candidate_files = 10
    args.prefilter_files = 80
    args.max_file_chars = 12000
    args.inspect_files = 10
    args.candidate_statements = 200

    load_dotenv()
    pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    code2vec = ExhaustiveCode2Vec(Path(args.checkpoint), Path(args.git_root), Path(args.vector_cache))
    file_pair_records = []
    file_pair_predictions: dict[tuple[int, str, str], list[str]] = {}
    code_snapshot_cache: dict[tuple[str, str], dict[str, str]] = {}
    ablation_rows = []
    errors = []

    for index, pair in enumerate(pairs, start=1):
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, oracle = split_detector_state(setup)
            split = split_label(pair)
            gt_files = set(oracle["target_changed_paths"])
            base = action_graphqa(state, 40)
            file_variants = {
                "Full PropQA": enhanced_file_rank(state, base, 10, 20),
                "w/o ExactCounterpart": enhanced_file_rank(
                    state, base, 10, 20, use_exact_counterpart=False
                ),
                "w/o per-source-file quota": enhanced_file_rank(
                    state, base, 10, 20, use_per_source_quota=False
                ),
            }
            for method, predictions in file_variants.items():
                ablation_rows.append(
                    {
                        "index": index,
                        "split": split,
                        "level": "patch_file",
                        "method": method,
                        **file_metrics([item["path"] for item in predictions], gt_files, 10),
                    }
                )

            target_paths = list(oracle["target_changed_paths"])
            blocks, intents = direct_intent_ast_blocks(state, runner, target_paths, 2)
            budget = min(1200, max(100, 8 * len(intents), 6 * len(target_paths)))
            full_statements = expand_direct_blocks(
                client, state, blocks, intents, budget, 6
            )
            statement_variants = {
                "Full PropQA": full_statements,
                "w/o InsertAnchor": [
                    item for item in full_statements if item.get("action") != "InsertAnchor"
                ],
                "w/o DeleteRegion": [
                    item for item in full_statements if item.get("action") != "DeleteRegion"
                ],
                "w/o intent-to-AST": lexical_statement_baseline(
                    state, target_paths, Path(args.git_root), budget
                ),
            }
            gt_statements = statement_ground_truth_from_files(oracle["target_files"])
            for method, predictions in statement_variants.items():
                ablation_rows.append(
                    {
                        "index": index,
                        "split": split,
                        "level": "statement",
                        "method": method,
                        **statement_metrics(predictions, gt_statements, 100, tolerance=2),
                    }
                )

            records = build_file_pair_records(index, pair, setup)
            file_pair_records.extend(records)
            snapshot_key = (state["target"].repo, state["target"].base_sha)
            if snapshot_key not in code_snapshot_cache:
                code_snapshot_cache[snapshot_key] = git_code_snapshot(
                    Path(args.git_root), state["target"].repo, state["target"].base_sha
                )
            code_snapshot = code_snapshot_cache[snapshot_key]
            for source_file in sorted({record["source_file"] for record in records}):
                source_state = restrict_state_to_source_file(state, source_file)
                graphqa = graphqa_action_files(source_state, SimpleNamespace(candidate_files=5))
                nicad = local_nicad_files(
                    source_state,
                    Path(args.git_root),
                    5,
                    code_snapshot,
                )
                c2v = code2vec.rank_files(
                    source_state["propagator"].repo,
                    source_state["propagator"].merged_sha,
                    [source_file],
                    source_state["target"].repo,
                    source_state["target"].base_sha,
                    5,
                )
                for method, predictions in (
                    ("Open-NiCad", nicad),
                    ("code2vec", c2v),
                    ("PropQA", graphqa),
                ):
                    file_pair_predictions[(index, source_file, method)] = [
                        item["path"] for item in predictions
                    ]
            print(f"[{index}/{len(pairs)}] ok", flush=True)
        except Exception as exc:
            errors.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)

    summaries = []
    for split in ["all", "simple", "hard"]:
        for level in ["patch_file", "statement"]:
            methods = sorted({row["method"] for row in ablation_rows if row["level"] == level})
            fields = ["precision", "recall", "f1", "mrr"] if level == "patch_file" else ["precision", "coverage", "f1", "mrr"]
            for method in methods:
                selected = [
                    row
                    for row in ablation_rows
                    if row["level"] == level
                    and row["method"] == method
                    and (split == "all" or row["split"] == split)
                ]
                summaries.append(
                    {
                        "split": split,
                        "level": level,
                        "method": method,
                        "pairs": len(selected),
                        **average(selected, fields),
                    }
                )

    file_pair_args = SimpleNamespace(candidate_files=5)
    import run_file_pair_detection_experiment as file_pair_module

    file_pair_module.args = file_pair_args
    file_pair_rows = metrics_for_records(
        file_pair_records,
        file_pair_predictions,
        ["Open-NiCad", "code2vec", "PropQA"],
        [1, 3, 5],
    )
    payload = {
        "dataset": args.dataset,
        "evaluated_pairs": len(pairs) - len(errors),
        "excluded": errors,
        "ablation_summary": summaries,
        "file_pair_records": len(file_pair_records),
        "file_pair_summary": file_pair_rows,
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
    for row in summaries:
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
    for row in file_pair_rows:
        if row["metric"] != "MeanRank":
            lines.append(f"| {row['method']} | {row['metric']} | {row['value']:.3f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
