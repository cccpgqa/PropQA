#!/usr/bin/env python3
"""Evaluate block-evidence-to-statement localization.

This experiment keeps the final evaluation unit as statements.  The detector
first aggregates PropQA/clone/code2vec statement evidence into target blocks,
then expands each target pre-fix block into ranked statement candidates.
Target changed statements are used only as the evaluation oracle.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from no_leak_ground_truth import StatementGT
from propagation_detector import GitHubClient, token_similarity
import run_block_first_statement_experiment as block_first
from run_block_first_statement_experiment import build_blocks, metrics, table, to_gt, unique_rank, weighted_line_rank
from run_llm_in_loop_hard_cases import prepare_state
from run_no_leak_five_module_pipeline import normalize_target_oracle_to_dataset_files, split_detector_state
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv, meaningful_statement


def stmt_key(item: dict[str, Any]) -> tuple[str, int] | None:
    try:
        return str(item["path"]), int(item["line"])
    except Exception:
        return None


def source_examples_for_block(block: dict[str, Any], limit: int) -> list[str]:
    examples = []
    seen = set()
    for item in block.get("items", []):
        text = (item.get("source_text") or "").strip()
        if text and text not in seen:
            examples.append(text)
            seen.add(text)
        if len(examples) >= limit:
            break
    return examples


def line_kind_score(source_examples: list[str], line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 0.0
    keywords = {
        "if": 0.08,
        "return": 0.07,
        "for": 0.05,
        "switch": 0.05,
        "case": 0.04,
        "delete": 0.04,
        "append": 0.04,
        "panic": 0.04,
        "error": 0.04,
    }
    score = 0.0
    lower = stripped.lower()
    for key, value in keywords.items():
        if key in lower:
            score += value
    if any(("return" in src.lower() or "error" in src.lower()) for src in source_examples) and ("return" in lower or "error" in lower):
        score += 0.08
    if any(("if " in src.lower() or src.strip().startswith("if")) for src in source_examples) and stripped.startswith("if"):
        score += 0.08
    return min(score, 0.25)


def proximity_score(line_no: int, seed_lines: list[int]) -> float:
    if not seed_lines:
        return 0.0
    distance = min(abs(line_no - seed) for seed in seed_lines)
    if distance == 0:
        return 1.0
    if distance <= 2:
        return 0.75
    if distance <= 5:
        return 0.45
    if distance <= 10:
        return 0.25
    return 0.0


def block_to_statement_agent(
    client: GitHubClient,
    repo: str,
    base_sha: str,
    blocks: list[dict[str, Any]],
    top: int,
    source_example_limit: int,
) -> list[dict[str, Any]]:
    ranked = []
    for block_rank, block in enumerate(blocks, start=1):
        content = client.file_at(repo, base_sha, block["path"]) or ""
        lines = content.splitlines()
        if not lines:
            continue
        start = max(1, int(block["start"]))
        end = min(len(lines), int(block["end"]))
        source_examples = source_examples_for_block(block, source_example_limit)
        seed_lines = [int(line) for line in block.get("candidate_lines", [])]
        for line_no in range(start, end + 1):
            text = lines[line_no - 1].strip()
            if not meaningful_statement(text):
                continue
            matched_source = max(source_examples, key=lambda src: token_similarity(src, text), default="")
            sim = token_similarity(matched_source, text) if matched_source else 0.0
            prox = proximity_score(line_no, seed_lines)
            kind = line_kind_score(source_examples, text)
            support = min(len(block.get("support_modules", [])), 3) / 3.0
            # QA-style scoring: first trust recalled block, then select
            # statements matching source intent or near seed evidence.
            score = (
                0.34 * float(block.get("score", 0.0) or 0.0)
                + 0.28 * sim
                + 0.20 * prox
                + 0.10 * support
                + 0.08 * kind
                + 0.04 / (block_rank + 1)
            )
            ranked.append(
                {
                    "path": block["path"],
                    "line": line_no,
                    "score": round(score, 6),
                    "line_similarity": round(sim, 6),
                    "proximity_score": round(prox, 6),
                    "line_text": text[:240],
                    "target_symbol": block.get("symbol"),
                    "block_start": start,
                    "block_end": end,
                    "block_score": block.get("score"),
                    "statement_modules": block.get("support_modules", []),
                    "matched_source_text": matched_source[:240],
                    "intent_key": matched_source[:120],
                    "refiner": "block_to_statement_agent",
                }
            )
    ranked.sort(key=lambda item: (-float(item["score"]), item["path"], int(item["line"])))
    return unique_rank(ranked, top)


def evidence_gated(preds: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    """Remove broad-block noise while retaining strong semantic/seed evidence."""
    kept = []
    for item in preds:
        sim = float(item.get("line_similarity", 0.0) or 0.0)
        prox = float(item.get("proximity_score", 0.0) or 0.0)
        modules = len(item.get("statement_modules", []))
        block_score = float(item.get("block_score", 0.0) or 0.0)
        if (
            sim >= 0.18
            or prox >= 0.75
            or (sim >= 0.10 and prox >= 0.45)
            or (modules >= 3 and block_score >= 0.9 and prox >= 0.25)
        ):
            kept.append(item)
    return unique_rank(kept, top)


def round_robin_by_intent(preds: list[dict[str, Any]], top: int, per_intent_cap: int = 8) -> list[dict[str, Any]]:
    """Preserve coverage across distinct source patch intents."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in preds:
        intent = item.get("intent_key") or item.get("matched_source_text") or f"{item['path']}:{item.get('target_symbol')}"
        groups[str(intent)].append(item)
    for key, items in groups.items():
        items.sort(key=lambda item: (-float(item["score"]), item["path"], int(item["line"])))
        groups[key] = items[:per_intent_cap]
    ordered_keys = sorted(groups, key=lambda key: -float(groups[key][0]["score"]) if groups[key] else 0.0)
    out = []
    while len(out) < top and ordered_keys:
        progressed = False
        for key in list(ordered_keys):
            if groups[key]:
                out.append(groups[key].pop(0))
                progressed = True
                if len(out) >= top:
                    break
            if not groups[key]:
                ordered_keys.remove(key)
        if not progressed:
            break
    return unique_rank(out, top)


def round_robin_by_block(preds: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in preds:
        groups[(item["path"], int(item.get("block_start", item["line"])), int(item.get("block_end", item["line"])))].append(item)
    for items in groups.values():
        items.sort(key=lambda item: (-float(item["score"]), item["path"], int(item["line"])))
    out = []
    while len(out) < top and groups:
        progressed = False
        for key in list(groups):
            if groups[key]:
                out.append(groups[key].pop(0))
                progressed = True
                if len(out) >= top:
                    break
            if not groups[key]:
                groups.pop(key, None)
        if not progressed:
            break
    return unique_rank(out, top)


def merge_ranks(*ranked_lists: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for ranked in ranked_lists:
        for item in ranked:
            key = stmt_key(item)
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(dict(item))
            if len(out) >= top:
                return out
    return out


def line_head_then_expand(line_rank: list[dict[str, Any]], expanded: list[dict[str, Any]], top: int, head: int) -> list[dict[str, Any]]:
    return merge_ranks(line_rank[:head], expanded, line_rank[head:], top=top)


def aggregate(rows: list[dict[str, Any]], method: str, split: str, k: int) -> dict[str, Any]:
    subset = [r for r in rows if r["method"] == method and r["dataset"] == split and r["k"] == k]
    out = {"method": method, "dataset": split, "k": k, "pairs": len(subset)}
    for key in ["precision", "coverage", "f1", "mrr"]:
        out[key] = sum(row[key] for row in subset) / len(subset) if subset else 0.0
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    module_map = {
        "graphqa": "stmt_graphqa",
        "clone": "stmt_text_clone",
        "code2vec": "stmt_code2vec",
    }
    selected_modules = tuple(module_map[name] for name in args.block_modules)
    old_base_modules = block_first.BASE_MODULES
    block_first.BASE_MODULES = selected_modules
    pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    strict = json.loads(Path(args.strict).read_text(encoding="utf-8"))
    strict_results = [r for r in strict.get("results", []) if r.get("status") == "ok"]
    if args.max_pairs:
        strict_results = strict_results[: args.max_pairs]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"), sleep_seconds=args.sleep)
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=args.statement_top)

    details = []
    rows = []
    try:
        for pos, saved in enumerate(strict_results, start=1):
            index = int(saved["index"])
            pair = pairs[index - 1]
            setup = prepare_state(runner, pair)
            if setup.get("status") and setup.get("status") != "ok":
                continue
            setup = normalize_target_oracle_to_dataset_files(setup)
            state, _oracle = split_detector_state(setup)
            split = saved.get("dataset_split") or "simple"
            gt_items = to_gt(saved.get("ground_truth", {}).get("statement_ground_truth") or [])
            modules = saved.get("statement_module_outputs") or {}
            line_rank = weighted_line_rank(modules, args.statement_top)
            blocks = build_blocks(modules, args.candidate_limit, args.block_radius)[: args.block_top]
            block_agent = block_to_statement_agent(
                client,
                state["target"].repo,
                state["target"].base_sha,
                blocks,
                args.statement_top,
                args.source_examples,
            )
            gated = evidence_gated(block_agent, args.statement_top)
            intent_rr = round_robin_by_intent(gated, args.statement_top, args.per_intent_cap)
            block_rr = round_robin_by_block(block_agent, args.statement_top)
            hybrid = merge_ranks(block_agent, line_rank, top=args.statement_top)
            rr_hybrid = merge_ranks(block_rr, line_rank, top=args.statement_top)
            intent_hybrid = merge_ranks(intent_rr, line_rank, top=args.statement_top)
            line20_intent = line_head_then_expand(line_rank, intent_rr, args.statement_top, 20)
            line50_intent = line_head_then_expand(line_rank, intent_rr, args.statement_top, 50)
            line20_gated = line_head_then_expand(line_rank, gated, args.statement_top, 20)
            suffix = "+".join(args.block_modules)
            methods = {
                "line-rank-no-dependency": line_rank,
                f"block-to-statement-agent[{suffix}]": block_agent,
                f"block-evidence-gated-agent[{suffix}]": gated,
                f"block-intent-round-robin-agent[{suffix}]": intent_rr,
                f"block-round-robin-agent[{suffix}]": block_rr,
                f"block-agent-plus-line-rank[{suffix}]": hybrid,
                f"block-rr-plus-line-rank[{suffix}]": rr_hybrid,
                f"block-intent-plus-line-rank[{suffix}]": intent_hybrid,
                f"line-head20-plus-block-intent[{suffix}]": line20_intent,
                f"line-head50-plus-block-intent[{suffix}]": line50_intent,
                f"line-head20-plus-block-gated[{suffix}]": line20_gated,
            }
            for method, preds in methods.items():
                for k in args.eval_ks:
                    rows.append({"index": index, "dataset": split, "method": method, "k": k, **metrics(preds, gt_items, k, args.tolerance)})
            details.append(
                {
                    "index": index,
                    "dataset": split,
                    "gt_statement_count": len(gt_items),
                    "prediction_counts": {method: len(preds) for method, preds in methods.items()},
                    "block_modules": list(args.block_modules),
                    "top_blocks": [
                        {
                            "path": b["path"],
                            "symbol": b["symbol"],
                            "start": b["start"],
                            "end": b["end"],
                            "score": b["score"],
                            "support_modules": b["support_modules"],
                        }
                        for b in blocks[:10]
                    ],
                    "block_agent_predictions": block_agent[:20],
                    "gated_predictions": gated[:20],
                    "intent_round_robin_predictions": intent_rr[:20],
                    "line_rank_predictions": line_rank[:20],
                }
            )
            print(f"[{pos}/{len(strict_results)}] index={index} split={split} blocks={len(blocks)}", flush=True)
    finally:
        block_first.BASE_MODULES = old_base_modules

    methods = sorted({row["method"] for row in rows})
    summary_rows = []
    for split in ["all", "simple", "hard"]:
        for method in methods:
            for k in args.eval_ks:
                if split == "all":
                    subset = [r for r in rows if r["method"] == method and r["k"] == k]
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
                    summary_rows.append(aggregate(rows, method, split, k))
    return {
        "summary": {
            "dataset": args.dataset,
            "strict": args.strict,
            "pairs": len(strict_results),
            "block_top": args.block_top,
            "block_radius": args.block_radius,
            "statement_top": args.statement_top,
            "block_modules": list(args.block_modules),
            "note": "Target changed statements are used only for evaluation.",
        },
        "rows": summary_rows,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/URL_Results_detection_primary_v4.json")
    parser.add_argument("--strict", default="outputs/strict_v4_semantic_aux_qwen_full.json")
    parser.add_argument("--output", default="outputs/block_to_statement_agent_experiment.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--statement-top", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--block-top", type=int, default=20)
    parser.add_argument("--block-radius", type=int, default=40)
    parser.add_argument("--source-examples", type=int, default=24)
    parser.add_argument("--per-intent-cap", type=int, default=8)
    parser.add_argument(
        "--block-modules",
        choices=["graphqa", "clone", "code2vec"],
        nargs="+",
        default=["graphqa", "clone", "code2vec"],
    )
    parser.add_argument("--eval-ks", type=int, nargs="*", default=[20, 50, 100])
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(
        "# Block-to-Statement Agent Experiment\n\n"
        + f"- Block Top: {payload['summary']['block_top']}\n"
        + f"- Block Radius: {payload['summary']['block_radius']}\n"
        + f"- Statement Top: {payload['summary']['statement_top']}\n"
        + f"- Block Modules: {', '.join(payload['summary']['block_modules'])}\n"
        + "- Final evaluation unit: statement.\n\n"
        + table(payload["rows"])
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
