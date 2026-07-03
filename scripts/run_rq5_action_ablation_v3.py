#!/usr/bin/env python3
"""Focused action ablations for file- and statement-level PropQA."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from no_leak_ground_truth import line_hits_statement_gt, statement_ground_truth_from_files
from propagation_detector import GitHubClient
from run_block_first_statement_experiment import metrics as statement_metrics
from run_file_pair_detection_experiment import (
    build_file_pair_records,
    graphqa_files,
    restrict_state_to_source_file,
)
from run_graphqa_action_enhancement_experiment import direct_intent_ast_blocks, expand_direct_blocks
from run_llm_in_loop_hard_cases import prepare_state
from run_no_leak_five_module_pipeline import normalize_target_oracle_to_dataset_files, split_detector_state
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv


FILE_VARIANTS = (
    "PropQA",
    "w/o ExactCounterpart",
    "w/o FindCounterpart",
    "w/o ProposeNewCounterpart",
)


def file_action_rank(
    state: dict[str, Any],
    top: int,
    *,
    use_exact: bool = True,
    use_find: bool = True,
    use_propose: bool = True,
) -> list[dict[str, Any]]:
    """Compose the final file-level actions for one source-file query.

    FindCounterpart is the structural/path retrieval action. ExactCounterpart
    and ProposeNewCounterpart add explicit source-path evidence when that path
    is present or absent, respectively.
    """
    source_path = state["source_changed_paths"][0]
    target_paths = set(state["target_tree"])
    ranked = graphqa_files(state, SimpleNamespace(candidate_files=top)) if use_find else []
    by_path = {item["path"]: dict(item) for item in ranked}

    promoted: list[dict[str, Any]] = []
    if use_exact and source_path in target_paths:
        item = by_path.pop(source_path, {"path": source_path, "matched_source_path": source_path})
        item.update({"score": 1.2, "action": "ExactCounterpart"})
        promoted.append(item)
    elif use_propose and source_path not in target_paths:
        promoted.append(
            {
                "path": source_path,
                "score": 1.2,
                "matched_source_path": source_path,
                "action": "ProposeNewCounterpart",
            }
        )

    return (promoted + list(by_path.values()))[:top]


def aggregate_statement(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    fields = ("precision", "coverage", "f1", "mrr")
    return {
        "pairs": len(rows),
        **{
            field: sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0
            for field in fields
        },
    }


def file_metrics_for_records(
    records: list[dict[str, Any]],
    predictions: dict[tuple[int, str, str], list[str]],
    methods: list[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        ranks = []
        for record in records:
            predicted = predictions.get((record["pair_index"], record["source_file"], method), [])
            rank = next(
                (position for position, path in enumerate(predicted, start=1) if path == record["target_file"]),
                None,
            )
            ranks.append(rank)
        for k in ks:
            value = (
                sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)
                if ranks
                else 0.0
            )
            rows.append({"method": method, "metric": f"Hit@{k}", "value": value, "file_pairs": len(ranks)})
        mrr = (
            sum(1.0 / rank for rank in ranks if rank is not None and rank <= 5) / len(ranks)
            if ranks
            else 0.0
        )
        rows.append({"method": method, "metric": "MRR@5", "value": mrr, "file_pairs": len(ranks)})
    return rows


def covered_gt(predictions: list[dict[str, Any]], gt_items: list[Any], top: int, tolerance: int) -> set[int]:
    covered: set[int] = set()
    for item in predictions[:top]:
        path = item.get("path")
        line = item.get("line")
        if path is None or line is None:
            continue
        for index, gt in enumerate(gt_items):
            if line_hits_statement_gt(str(path), int(line), gt, tolerance):
                covered.add(index)
    return covered


def format_file_table(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    counts: dict[str, int] = {}
    for row in rows:
        grouped[row["method"]][row["metric"]] = float(row["value"])
        counts[row["method"]] = int(row["file_pairs"])
    lines = [
        "| Variant | File pairs | Hit@1 | Hit@3 | Hit@5 | MRR@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in FILE_VARIANTS:
        values = grouped[method]
        lines.append(
            f"| {method} | {counts[method]} | {values['Hit@1']:.3f} | "
            f"{values['Hit@3']:.3f} | {values['Hit@5']:.3f} | {values['MRR@5']:.3f} |"
        )
    return "\n".join(lines)


def format_statement_table(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Ground truth | Variant | Pairs | Precision@100 | Coverage@100 | F1@100 | MRR@100 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for kind in ("all", "added_new", "edited", "deleted"):
        for method in ("PropQA", "w/o InsertAnchor"):
            row = summary[kind][method]
            lines.append(
                f"| {kind} | {method} | {row['pairs']} | {row['precision']:.3f} | "
                f"{row['coverage']:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/URL_Results_detection_url_celo_v2.json")
    parser.add_argument("--output", default="outputs/rq5_action_ablation_v3.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--file-top", type=int, default=5)
    parser.add_argument("--statement-top", type=int, default=100)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--checkpoint", default=".cache/rq5_action_ablation_v3.pkl")
    args = parser.parse_args()

    load_dotenv()
    pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.max_pairs:
        saved = pickle.loads(checkpoint.read_bytes())
        records = saved["records"]
        file_predictions = saved["file_predictions"]
        statement_rows = saved["statement_rows"]
        insert_counts = saved["insert_counts"]
        errors = saved["errors"]
        completed = int(saved["completed"])
        print(f"resuming after pair {completed}", flush=True)
    else:
        records: list[dict[str, Any]] = []
        file_predictions: dict[tuple[int, str, str], list[str]] = {}
        statement_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
            kind: {"PropQA": [], "w/o InsertAnchor": []}
            for kind in ("all", "added_new", "edited", "deleted")
        }
        insert_counts = {
            "added_new_ground_truth": 0,
            "covered_by_graphqa": 0,
            "covered_without_insert_anchor": 0,
            "covered_only_with_insert_anchor": 0,
        }
        errors = []
        completed = 0

    for pair_index, pair in enumerate(pairs, start=1):
        if pair_index <= completed:
            continue
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, oracle = split_detector_state(setup)

            pair_records = build_file_pair_records(pair_index, pair, setup)
            records.extend(pair_records)
            for source_file in sorted({record["source_file"] for record in pair_records}):
                source_state = restrict_state_to_source_file(state, source_file)
                variants = {
                    "PropQA": file_action_rank(source_state, args.file_top),
                    "w/o ExactCounterpart": file_action_rank(
                        source_state, args.file_top, use_exact=False
                    ),
                    "w/o FindCounterpart": file_action_rank(
                        source_state, args.file_top, use_find=False
                    ),
                    "w/o ProposeNewCounterpart": file_action_rank(
                        source_state, args.file_top, use_propose=False
                    ),
                }
                for method, ranked in variants.items():
                    file_predictions[(pair_index, source_file, method)] = [
                        item["path"] for item in ranked
                    ]

            target_paths = list(oracle["target_changed_paths"])
            blocks, intents = direct_intent_ast_blocks(state, runner, target_paths, 2)
            budget = min(1200, max(args.statement_top, 8 * len(intents), 6 * len(target_paths)))
            full = expand_direct_blocks(
                client,
                state,
                blocks,
                intents,
                budget,
                6,
                use_insert_anchor=True,
            )
            without_insert = expand_direct_blocks(
                client,
                state,
                blocks,
                intents,
                budget,
                6,
                use_insert_anchor=False,
            )
            gt_all = statement_ground_truth_from_files(oracle["target_files"])
            for kind in ("all", "added_new", "edited", "deleted"):
                gt = gt_all if kind == "all" else [item for item in gt_all if item.kind == kind]
                if not gt:
                    continue
                statement_rows[kind]["PropQA"].append(
                    statement_metrics(full, gt, args.statement_top, args.tolerance)
                )
                statement_rows[kind]["w/o InsertAnchor"].append(
                    statement_metrics(without_insert, gt, args.statement_top, args.tolerance)
                )

            added = [item for item in gt_all if item.kind == "added_new"]
            full_covered = covered_gt(full, added, args.statement_top, args.tolerance)
            without_covered = covered_gt(without_insert, added, args.statement_top, args.tolerance)
            insert_counts["added_new_ground_truth"] += len(added)
            insert_counts["covered_by_graphqa"] += len(full_covered)
            insert_counts["covered_without_insert_anchor"] += len(without_covered)
            insert_counts["covered_only_with_insert_anchor"] += len(full_covered - without_covered)
            print(f"[{pair_index}/{len(pairs)}] ok", flush=True)
        except Exception as exc:
            errors.append({"index": pair_index, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{pair_index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)
        if not args.max_pairs and (pair_index % 5 == 0 or pair_index == len(pairs)):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(
                pickle.dumps(
                    {
                        "completed": pair_index,
                        "records": records,
                        "file_predictions": file_predictions,
                        "statement_rows": statement_rows,
                        "insert_counts": insert_counts,
                        "errors": errors,
                    }
                )
            )

    file_rows = file_metrics_for_records(records, file_predictions, list(FILE_VARIANTS))
    statement_summary = {
        kind: {
            method: aggregate_statement(rows)
            for method, rows in methods.items()
        }
        for kind, methods in statement_rows.items()
    }
    payload = {
        "dataset": args.dataset,
        "evaluated_pairs": len(pairs) - len(errors),
        "file_pair_records": len(records),
        "file_ablation": file_rows,
        "statement_insert_anchor_ablation": statement_summary,
        "insert_anchor_added_new_counts": insert_counts,
        "errors": errors,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = (
        "# RQ5 Action Ablation V3\n\n"
        "## File-level leave-one-action-out\n\n"
        + format_file_table(file_rows)
        + "\n\n## InsertAnchor operation-specific ablation\n\n"
        + format_statement_table(statement_summary)
        + "\n\n## Added-new coverage counts\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in insert_counts.items())
        + "\n"
    )
    output.with_suffix(".md").write_text(md, encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(f"wrote {output}", flush=True)
    print(f"wrote {output.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
