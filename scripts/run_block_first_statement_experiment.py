#!/usr/bin/env python3
"""Evaluate block/function-first statement localization variants.

This is an offline no-leak experiment over saved statement candidates.  It does
not use target PR/commit metadata; target changed statements are loaded only as
the evaluation oracle.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from no_leak_ground_truth import StatementGT, line_hits_statement_gt


BASE_MODULES = ("stmt_graphqa", "stmt_text_clone", "stmt_code2vec")
MODULE_WEIGHTS = {"stmt_graphqa": 1.18, "stmt_text_clone": 1.10, "stmt_code2vec": 1.00}


def to_gt(items: list[dict[str, Any]]) -> list[StatementGT]:
    fields = StatementGT.__dataclass_fields__
    return [StatementGT(**{field: item.get(field) for field in fields}) for item in items]


def stmt_key(item: dict[str, Any]) -> tuple[str, int] | None:
    try:
        return str(item["path"]), int(item["line"])
    except Exception:
        return None


def unique_rank(items: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in items:
        key = stmt_key(item)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= top:
            break
    return out


def weighted_line_rank(modules: dict[str, list[dict[str, Any]]], top: int) -> list[dict[str, Any]]:
    scores: dict[tuple[str, int], float] = defaultdict(float)
    items: dict[tuple[str, int], dict[str, Any]] = {}
    support: dict[tuple[str, int], set[str]] = defaultdict(set)
    for module in BASE_MODULES:
        for rank, item in enumerate(modules.get(module, [])[:top], start=1):
            key = stmt_key(item)
            if key is None:
                continue
            score = MODULE_WEIGHTS[module] * (float(item.get("score", 0.0) or 0.0) + 1.0 / (rank + 2))
            scores[key] += score
            support[key].add(module)
            items.setdefault(key, dict(item))
    ranked = []
    for key, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        item = dict(items[key])
        item["score"] = round(score + 0.05 * len(support[key]), 6)
        item["statement_modules"] = sorted(support[key])
        ranked.append(item)
    return ranked[:top]


def block_key(item: dict[str, Any]) -> tuple[str, str]:
    symbol = item.get("target_symbol") or ""
    if not symbol:
        # Keep nearby non-symbol candidates separate by coarse line buckets.
        line = int(item.get("line") or 0)
        symbol = f"line_bucket_{line // 25}"
    return str(item.get("path") or ""), str(symbol)


def build_blocks(modules: dict[str, list[dict[str, Any]]], candidate_limit: int, radius: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    module_support: dict[tuple[str, str], set[str]] = defaultdict(set)
    for module in BASE_MODULES:
        for item in modules.get(module, [])[:candidate_limit]:
            key = stmt_key(item)
            if key is None:
                continue
            bkey = block_key(item)
            copied = dict(item)
            copied["_module"] = module
            copied["_rank_weight"] = 1.0 / (len(grouped[bkey]) + 3)
            grouped[bkey].append(copied)
            module_support[bkey].add(module)

    blocks = []
    for bkey, items in grouped.items():
        path, symbol = bkey
        lines = sorted(int(item["line"]) for item in items)
        max_score = max(float(item.get("score", 0.0) or 0.0) for item in items)
        avg_score = sum(float(item.get("score", 0.0) or 0.0) for item in items) / len(items)
        support = module_support[bkey]
        start = max(1, min(lines) - radius)
        end = max(lines) + radius
        density = min(len(items), 12) / 12.0
        score = 0.50 * max_score + 0.22 * avg_score + 0.18 * len(support) + 0.10 * density
        blocks.append(
            {
                "path": path,
                "symbol": symbol,
                "start": start,
                "end": end,
                "score": round(score, 6),
                "support_modules": sorted(support),
                "candidate_lines": lines,
                "items": items,
            }
        )
    blocks.sort(key=lambda b: (-float(b["score"]), b["path"], int(b["start"])))
    return blocks


def refine_block_to_statements(blocks: list[dict[str, Any]], top: int, radius: int, stride: int) -> list[dict[str, Any]]:
    refined = []
    for block_rank, block in enumerate(blocks, start=1):
        item_by_line: dict[int, dict[str, Any]] = {}
        for item in sorted(block["items"], key=lambda x: -float(x.get("score", 0.0) or 0.0)):
            item_by_line.setdefault(int(item["line"]), item)

        # First keep concrete candidate statements inside the block.
        for line, item in sorted(item_by_line.items()):
            out = dict(item)
            out["score"] = round(float(block["score"]) + float(item.get("score", 0.0) or 0.0) + 1.0 / (block_rank + 2), 6)
            out["block_start"] = block["start"]
            out["block_end"] = block["end"]
            out["block_symbol"] = block["symbol"]
            out["block_refinement"] = "candidate_line"
            refined.append(out)

        # Then probe nearby anchors within the block.  This simulates the
        # second-stage QA/AST inspection choosing lines around a recalled block.
        probes = set()
        for line in block["candidate_lines"]:
            for delta in range(-radius, radius + 1, stride):
                probe = int(line) + delta
                if block["start"] <= probe <= block["end"]:
                    probes.add(probe)
        midpoint = (int(block["start"]) + int(block["end"])) // 2
        probes.add(midpoint)
        for line in sorted(probes):
            if line in item_by_line:
                continue
            refined.append(
                {
                    "path": block["path"],
                    "line": line,
                    "score": round(float(block["score"]) + 0.15 / (block_rank + 1), 6),
                    "line_text": "",
                    "target_symbol": block["symbol"],
                    "statement_modules": block["support_modules"],
                    "block_start": block["start"],
                    "block_end": block["end"],
                    "block_symbol": block["symbol"],
                    "block_refinement": "context_probe",
                }
            )
    refined.sort(key=lambda x: (-float(x.get("score", 0.0) or 0.0), str(x["path"]), int(x["line"])))
    return unique_rank(refined, top)


def block_refine_rank(modules: dict[str, list[dict[str, Any]]], top: int, candidate_limit: int, radius: int, stride: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = build_blocks(modules, candidate_limit, radius)
    return refine_block_to_statements(blocks, top, radius, stride), blocks


def metrics(predicted: list[dict[str, Any]], gt_items: list[StatementGT], k: int, tolerance: int) -> dict[str, float]:
    preds = predicted[:k]
    hit_pred = 0
    covered = set()
    rr = 0.0
    for rank, item in enumerate(preds, start=1):
        key = stmt_key(item)
        if key is None:
            continue
        path, line = key
        matched = [idx for idx, gt in enumerate(gt_items) if line_hits_statement_gt(path, line, gt, tolerance)]
        if matched:
            hit_pred += 1
            covered.update(matched)
            if rr == 0.0:
                rr = 1.0 / rank
    precision = hit_pred / len(preds) if preds else 0.0
    coverage = len(covered) / len(gt_items) if gt_items else 0.0
    f1 = 2 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
    return {"precision": precision, "coverage": coverage, "f1": f1, "mrr": rr}


def block_metrics(blocks: list[dict[str, Any]], gt_items: list[StatementGT], k: int, tolerance: int) -> dict[str, float]:
    chosen = blocks[:k]
    covered = set()
    rr = 0.0
    for rank, block in enumerate(chosen, start=1):
        hit = False
        for idx, gt in enumerate(gt_items):
            if block["path"] == gt.path and int(block["start"]) - tolerance <= gt.anchor_line <= int(block["end"]) + tolerance:
                covered.add(idx)
                hit = True
        if hit and rr == 0.0:
            rr = 1.0 / rank
    coverage = len(covered) / len(gt_items) if gt_items else 0.0
    # Block precision is coarse: a block is positive if it covers at least one GT anchor.
    positive_blocks = 0
    for block in chosen:
        if any(block["path"] == gt.path and int(block["start"]) - tolerance <= gt.anchor_line <= int(block["end"]) + tolerance for gt in gt_items):
            positive_blocks += 1
    precision = positive_blocks / len(chosen) if chosen else 0.0
    f1 = 2 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
    return {"precision": precision, "coverage": coverage, "f1": f1, "mrr": rr}


def aggregate(rows: list[dict[str, Any]], method: str, split: str, k: int) -> dict[str, Any]:
    subset = [r for r in rows if r["method"] == method and r["dataset"] == split and r["k"] == k]
    out = {"method": method, "dataset": split, "k": k, "pairs": len(subset)}
    for key in ["precision", "coverage", "f1", "mrr"]:
        out[key] = sum(row[key] for row in subset) / len(subset) if subset else 0.0
    return out


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Method | K | Pairs | Precision | Coverage | F1 | MRR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['k']} | {row['pairs']} | "
            f"{row['precision']:.3f} | {row['coverage']:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |"
        )
    return "\n".join(lines)


def block_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Block Variant | Top Blocks | Pairs | Block Precision | Block Coverage | Block F1 | MRR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['k']} | {row['pairs']} | "
            f"{row['precision']:.3f} | {row['coverage']:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    strict = json.loads(Path(args.strict).read_text(encoding="utf-8"))
    results = [r for r in strict.get("results", []) if r.get("status") == "ok"]
    if args.max_pairs:
        results = results[: args.max_pairs]

    methods = {
        "line-rank-no-dependency": lambda modules: (weighted_line_rank(modules, args.statement_top), []),
        "block-first-r5-s2": lambda modules: block_refine_rank(modules, args.statement_top, args.candidate_limit, 5, 2),
        "block-first-r8-s2": lambda modules: block_refine_rank(modules, args.statement_top, args.candidate_limit, 8, 2),
        "block-first-r12-s3": lambda modules: block_refine_rank(modules, args.statement_top, args.candidate_limit, 12, 3),
        "block-first-r20-s4": lambda modules: block_refine_rank(modules, args.statement_top, args.candidate_limit, 20, 4),
        "block-first-r40-s5": lambda modules: block_refine_rank(modules, args.statement_top, args.candidate_limit, 40, 5),
    }
    row_details = []
    block_row_details = []
    case_details = []
    for saved in results:
        gt_items = to_gt(saved.get("ground_truth", {}).get("statement_ground_truth") or [])
        split = saved.get("dataset_split") or "all"
        modules = saved.get("statement_module_outputs") or {}
        for method, fn in methods.items():
            preds, blocks = fn(modules)
            for k in args.eval_ks:
                row_details.append(
                    {
                        "index": saved["index"],
                        "dataset": split,
                        "method": method,
                        "k": k,
                        **metrics(preds, gt_items, k, args.tolerance),
                    }
                )
            if method.startswith("block-first"):
                case_details.append(
                    {
                        "index": saved["index"],
                        "dataset": split,
                        "method": method,
                        "gt_statement_count": len(gt_items),
                        "top_blocks": [
                            {
                                "path": b["path"],
                                "symbol": b["symbol"],
                                "start": b["start"],
                                "end": b["end"],
                                "score": b["score"],
                                "support_modules": b["support_modules"],
                            }
                            for b in blocks[:5]
                        ],
                        "top_predictions": preds[:10],
                    }
                )
                radius = int(method.split("-r", 1)[1].split("-s", 1)[0])
                blocks = build_blocks(modules, args.candidate_limit, radius)
                for k in args.block_eval_ks:
                    block_row_details.append(
                        {
                            "index": saved["index"],
                            "dataset": split,
                            "method": method,
                            "k": k,
                            **block_metrics(blocks, gt_items, k, args.tolerance),
                        }
                    )

    summary_rows = []
    for split in ["all", "simple", "hard"]:
        for method in methods:
            for k in args.eval_ks:
                if split == "all":
                    subset = [r for r in row_details if r["method"] == method and r["k"] == k]
                    summary_rows.append(
                        {
                            "method": method,
                            "dataset": split,
                            "k": k,
                            "pairs": len(subset),
                            **{
                                key: sum(row[key] for row in subset) / len(subset) if subset else 0.0
                                for key in ["precision", "coverage", "f1", "mrr"]
                            },
                        }
                    )
                else:
                    summary_rows.append(aggregate(row_details, method, split, k))
    block_summary_rows = []
    for split in ["all", "simple", "hard"]:
        for method in [m for m in methods if m.startswith("block-first")]:
            for k in args.block_eval_ks:
                if split == "all":
                    subset = [r for r in block_row_details if r["method"] == method and r["k"] == k]
                    block_summary_rows.append(
                        {
                            "method": method,
                            "dataset": split,
                            "k": k,
                            "pairs": len(subset),
                            **{
                                key: sum(row[key] for row in subset) / len(subset) if subset else 0.0
                                for key in ["precision", "coverage", "f1", "mrr"]
                            },
                        }
                    )
                else:
                    block_summary_rows.append(aggregate(block_row_details, method, split, k))
    return {
        "summary": {
            "strict": args.strict,
            "pairs": len(results),
            "base_modules": list(BASE_MODULES),
            "dependency_excluded": True,
            "note": "Target changed statements are used only for evaluation.",
        },
        "rows": summary_rows,
        "block_upper_bound_rows": block_summary_rows,
        "details": row_details,
        "block_details": block_row_details,
        "case_details": case_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", default="outputs/strict_v4_semantic_aux_qwen_full.json")
    parser.add_argument("--output", default="outputs/block_first_statement_experiment.json")
    parser.add_argument("--statement-top", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--eval-ks", type=int, nargs="*", default=[20, 50, 100])
    parser.add_argument("--block-eval-ks", type=int, nargs="*", default=[5, 10, 20, 50])
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(
        "# Block-First Statement Localization Experiment\n\n"
        + "- Dependency statement module excluded.\n"
        + "- Evaluation uses target statements only as oracle.\n\n"
        + "## Refined Statement Predictions\n\n"
        + table(payload["rows"])
        + "\n\n## Block-Level Upper Bound\n\n"
        + block_table(payload["block_upper_bound_rows"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(table(payload["rows"]))
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
