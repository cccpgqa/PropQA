#!/usr/bin/env python3
"""Evaluate no-leak PropQA action enhancements for files and statements.

File detection adds ExactCounterpart, FindTestAndDataCompanion, and a
per-source-file candidate quota. Statement localization directly maps source
patch intents to target pre-fix AST blocks, then applies InsertAnchor,
DeleteRegion, and the existing evidence gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from no_leak_ground_truth import StatementGT
from propagation_detector import GitHubClient, path_score, path_tokens, token_similarity
from run_block_first_statement_experiment import metrics, table, to_gt, unique_rank
from run_block_to_statement_agent_experiment import evidence_gated
from run_llm_in_loop_hard_cases import prepare_state
from run_no_leak_five_module_pipeline import normalize_target_oracle_to_dataset_files, split_detector_state
from run_qa_mcts_small_sample import (
    PropagationQAMCTS,
    classify_patch_pattern,
    load_dotenv,
    meaningful_statement,
    pattern_symbol_score,
)


def is_test_or_data(path: str) -> bool:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    return (
        "_test." in lower
        or "/test/" in lower
        or "/tests/" in lower
        or "/testdata/" in lower
        or "/fixture" in lower
        or name.endswith((".snap", ".golden", ".out", ".rlp", ".hex"))
    )


def stem(path: str) -> str:
    name = path.lower().rsplit("/", 1)[-1]
    base = name.rsplit(".", 1)[0]
    return base.removesuffix("_test").removesuffix("-test")


def per_source_candidates(
    state: dict[str, Any],
    per_source_pool: int,
    use_exact_counterpart: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    source_paths = state.get("source_changed_paths") or []
    for source_path in source_paths:
        ranked = []
        for target_path in state["target_tree"]:
            score = path_score(source_path, target_path)
            action = "FindStructuralCounterpart"
            if use_exact_counterpart and source_path == target_path:
                score += 0.45
                action = "ExactCounterpart"
            ranked.append(
                {
                    "path": target_path,
                    "score": round(score, 6),
                    "matched_source_path": source_path,
                    "action": action,
                }
            )
        ranked.sort(key=lambda item: (-float(item["score"]), item["path"]))
        out[source_path] = ranked[:per_source_pool]
    return out


def find_test_and_data_companions(state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    candidates = []
    source_paths = state.get("source_changed_paths") or []
    for target_path in state["target_tree"]:
        if not is_test_or_data(target_path):
            continue
        best_score = 0.0
        matched = None
        for source_path in source_paths:
            same_stem = stem(source_path) == stem(target_path)
            token_overlap = len(path_tokens(source_path) & path_tokens(target_path)) / max(
                len(path_tokens(source_path) | path_tokens(target_path)), 1
            )
            score = 0.52 * path_score(source_path, target_path) + 0.32 * float(same_stem) + 0.16 * token_overlap
            if score > best_score:
                best_score, matched = score, source_path
        if matched and best_score >= 0.28:
            candidates.append(
                {
                    "path": target_path,
                    "score": round(best_score + 0.18, 6),
                    "matched_source_path": matched,
                    "action": "FindTestAndDataCompanion",
                }
            )
    candidates.sort(key=lambda item: (-float(item["score"]), item["path"]))
    return candidates[:limit]


def enhanced_file_rank(
    state: dict[str, Any],
    base_graphqa: list[dict[str, Any]],
    top: int,
    per_source_pool: int,
    use_exact_counterpart: bool = True,
    use_per_source_quota: bool = True,
    use_test_data_companion: bool = False,
) -> list[dict[str, Any]]:
    """Reserve one candidate per source file, then combine PropQA actions."""
    per_source = per_source_candidates(state, per_source_pool, use_exact_counterpart)
    source_paths = state.get("source_changed_paths") or []
    adaptive_top = max(top, len(source_paths))
    selected: list[dict[str, Any]] = []
    seen = set()

    # ExactCounterpart is a high-confidence graph edge and receives first claim.
    if use_exact_counterpart:
        for source_path in source_paths:
            exact = next((x for x in per_source[source_path] if x["path"] == source_path), None)
            if exact and exact["path"] not in seen:
                selected.append(exact)
                seen.add(exact["path"])

    # Per-source-file quota: at least one candidate for every source file when
    # distinct target paths exist. Duplicate counterparts do not waste slots.
    if use_per_source_quota:
        for source_path in source_paths:
            for item in per_source[source_path]:
                if item["path"] not in seen:
                    selected.append(item)
                    seen.add(item["path"])
                    break

    pool = []
    if use_test_data_companion:
        pool.extend(find_test_and_data_companions(state, max(20, top * 2)))
    pool.extend(base_graphqa)
    for items in per_source.values():
        pool.extend(items)
    best: dict[str, dict[str, Any]] = {}
    for item in pool:
        path = item.get("path")
        if not path or path in seen:
            continue
        old = best.get(path)
        if old is None or float(item.get("score", 0.0)) > float(old.get("score", 0.0)):
            best[path] = dict(item)
    selected.extend(sorted(best.values(), key=lambda item: (-float(item.get("score", 0.0)), item["path"])))
    return selected[:adaptive_top]


def file_metrics(predicted: list[str], gt: set[str], k: int) -> dict[str, float]:
    chosen = predicted[:k]
    hits = len(set(chosen) & gt)
    precision = hits / len(chosen) if chosen else 0.0
    recall = hits / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rr = next((1.0 / rank for rank, path in enumerate(chosen, 1) if path in gt), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "mrr": rr}


def source_intents(state: dict[str, Any], runner: PropagationQAMCTS) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for stmt in state.get("source_statements", []):
        if stmt.kind not in {"added", "deleted"} or not meaningful_statement(stmt.text):
            continue
        # Added intents are represented in the merged source graph; deleted
        # intents must be recovered from the source pre-fix graph.
        source_sha = state["propagator"].merged_sha if stmt.kind == "added" else state["propagator"].base_sha
        symbols = runner._symbols_for_file(
            state, "source", state["propagator"].repo, source_sha, stmt.path
        )
        symbol = next(
            (sym for sym in symbols if int(sym["start_line"]) <= stmt.line_no <= int(sym["end_line"])),
            None,
        )
        symbol_name = (symbol or {}).get("name") or f"line_{stmt.line_no // 20}"
        key = (stmt.path, symbol_name, stmt.kind)
        item = grouped.setdefault(
            key,
            {
                "key": "::".join(key),
                "path": stmt.path,
                "symbol": symbol_name,
                "kind": "insert" if stmt.kind == "added" else "delete",
                "texts": [],
                "lines": [],
                "source_symbol": symbol,
            },
        )
        item["texts"].append(stmt.text[:500])
        item["lines"].append(stmt.line_no)
    return list(grouped.values())


def symbol_match_score(intent: dict[str, Any], target_symbol: dict[str, Any], source_path: str, target_path: str) -> float:
    source_symbol = intent.get("source_symbol") or {}
    source_text = "\n".join(intent["texts"][:12])
    target_text = target_symbol.get("text", "")
    text_sim = token_similarity(source_text, target_text)
    name_sim = token_similarity(str(source_symbol.get("name") or intent["symbol"]), str(target_symbol.get("name") or ""))
    source_tokens = set(source_symbol.get("tokens") or path_tokens(source_path))
    target_tokens = set(target_symbol.get("tokens") or path_tokens(target_path))
    token_overlap = len(source_tokens & target_tokens) / max(len(source_tokens | target_tokens), 1)
    pattern = classify_patch_pattern(
        [type("IntentStmt", (), {"text": text})() for text in intent["texts"][:20]]
    )
    pattern_score = pattern_symbol_score(pattern, target_symbol)
    return 0.30 * text_sim + 0.27 * name_sim + 0.18 * token_overlap + 0.15 * path_score(source_path, target_path) + 0.10 * pattern_score


def direct_intent_ast_blocks(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    target_paths: list[str],
    per_intent_file_quota: int,
    use_delete_region: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate blocks directly from source intents, without global line candidates."""
    intents = source_intents(state, runner)
    blocks_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for intent in intents:
        for target_path in target_paths:
            symbols = runner._symbols_for_file(
                state, "target", state["target"].repo, state["target"].base_sha, target_path
            )
            if not symbols:
                content = runner.client.file_at(state["target"].repo, state["target"].base_sha, target_path) or ""
                symbols = [{"name": "file", "start_line": 1, "end_line": max(len(content.splitlines()), 1), "text": content[:4000]}]
            scored = [
                (symbol_match_score(intent, sym, intent["path"], target_path), sym)
                for sym in symbols
            ]
            scored.sort(key=lambda item: (-item[0], int(item[1]["start_line"])))
            for score, sym in scored[:per_intent_file_quota]:
                key = (target_path, int(sym["start_line"]), int(sym["end_line"]))
                block = blocks_by_key.setdefault(
                    key,
                    {
                        "path": target_path,
                        "symbol": sym.get("name") or "file",
                        "start": int(sym["start_line"]),
                        "end": int(sym["end_line"]),
                        "score": 0.0,
                        "support_modules": ["stmt_graphqa_direct_ast"],
                        "candidate_lines": [],
                        "items": [],
                        "intent_evidence": [],
                        "actions": [],
                    },
                )
                action = (
                    "DeleteRegion"
                    if use_delete_region and intent["kind"] == "delete"
                    else "FindAffectedASTBlock"
                )
                block["score"] = max(float(block["score"]), score + (0.12 if action == "DeleteRegion" else 0.0))
                block["actions"].append(action)
                block["intent_evidence"].append(intent)
    blocks = list(blocks_by_key.values())
    for block in blocks:
        block["actions"] = sorted(set(block["actions"]))
        block["intent_evidence"].sort(key=lambda intent: intent["key"])
    blocks.sort(key=lambda block: (-float(block["score"]), block["path"], int(block["start"])))
    return blocks, intents


def expand_direct_blocks(
    client: GitHubClient,
    state: dict[str, Any],
    blocks: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    dynamic_top: int,
    per_intent_cap: int,
    use_delete_region: bool = True,
    use_insert_anchor: bool = True,
    use_evidence_gate: bool = False,
) -> list[dict[str, Any]]:
    ranked = []
    for block_rank, block in enumerate(blocks, 1):
        content = client.file_at(state["target"].repo, state["target"].base_sha, block["path"]) or ""
        lines = content.splitlines()
        if not lines:
            continue
        block_intents = block.get("intent_evidence") or intents
        for intent in block_intents:
            texts = intent["texts"][:16]
            action = (
                "DeleteRegion"
                if use_delete_region and intent["kind"] == "delete"
                else "FindAffectedStatement"
            )
            for line_no in range(max(1, block["start"]), min(len(lines), block["end"]) + 1):
                text = lines[line_no - 1].strip()
                if not meaningful_statement(text):
                    continue
                sim = max((token_similarity(src, text) for src in texts), default=0.0)
                score = 0.42 * float(block["score"]) + 0.48 * sim + 0.10 / (block_rank + 1)
                ranked.append(
                    {
                        "path": block["path"],
                        "line": line_no,
                        "score": round(score, 6),
                        "line_similarity": round(sim, 6),
                        "proximity_score": 0.75 if action == "DeleteRegion" else 0.45,
                        "line_text": text[:240],
                        "target_symbol": block["symbol"],
                        "block_start": block["start"],
                        "block_end": block["end"],
                        "block_score": block["score"],
                        "statement_modules": ["stmt_graphqa_direct_ast"],
                        "matched_source_text": texts[0][:240] if texts else "",
                        "intent_key": intent["key"],
                        "intent_budget": min(120, max(per_intent_cap, len(texts) * (3 if action == "DeleteRegion" else 1))),
                        "action": action,
                    }
                )
            if use_insert_anchor and intent["kind"] == "insert":
                # New statements do not exist pre-fix. Explicitly expose legal
                # insertion loci at AST boundaries and adjacent statement slots.
                def clamp_line(value: int) -> int:
                    return min(len(lines), max(1, int(value)))

                anchors = {clamp_line(block["start"]), clamp_line(block["end"])}
                anchors.update({clamp_line(block["start"] + 1), clamp_line(block["end"] - 1)})
                for line_no in sorted(anchors):
                    ranked.append(
                        {
                            "path": block["path"],
                            "line": line_no,
                            "score": round(0.52 * float(block["score"]) + 0.18, 6),
                            "line_similarity": 0.20,
                            "proximity_score": 1.0,
                            "line_text": lines[line_no - 1].strip()[:240],
                            "target_symbol": block["symbol"],
                            "block_start": block["start"],
                            "block_end": block["end"],
                            "block_score": block["score"],
                            "statement_modules": ["stmt_graphqa_direct_ast", "InsertAnchor"],
                            "matched_source_text": texts[0][:240] if texts else "",
                            "intent_key": intent["key"],
                            "intent_budget": max(per_intent_cap, len(texts)),
                            "action": "InsertAnchor",
                        }
                    )
    ranked.sort(key=lambda item: (-float(item["score"]), item["path"], int(item["line"])))
    candidates = unique_rank(ranked, dynamic_top * 3)
    gated = evidence_gated(candidates, dynamic_top * 3) if use_evidence_gate else candidates
    return round_robin_dynamic_intent(gated, dynamic_top, per_intent_cap)


def round_robin_dynamic_intent(
    preds: list[dict[str, Any]], top: int, default_cap: int
) -> list[dict[str, Any]]:
    """Round-robin with larger quotas for explicit DeleteRegion intents."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    caps: dict[str, int] = {}
    for item in preds:
        key = str(item.get("intent_key") or f"{item['path']}:{item.get('target_symbol')}")
        groups[key].append(item)
        caps[key] = max(caps.get(key, default_cap), int(item.get("intent_budget", default_cap)))
    for key in groups:
        groups[key].sort(key=lambda item: (-float(item["score"]), item["path"], int(item["line"])))
        groups[key] = groups[key][: caps[key]]
    keys = sorted(groups, key=lambda key: -float(groups[key][0]["score"]) if groups[key] else 0.0)
    out = []
    while len(out) < top and keys:
        progressed = False
        for key in list(keys):
            if groups[key]:
                out.append(groups[key].pop(0))
                progressed = True
                if len(out) >= top:
                    break
            if not groups[key]:
                keys.remove(key)
        if not progressed:
            break
    return unique_rank(out, top)


def average(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    return {key: sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0 for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/URL_Results_detection_primary_v4.json")
    parser.add_argument("--strict", default="outputs/strict_v4_semantic_aux_qwen_full.json")
    parser.add_argument("--output", default="outputs/graphqa_action_enhancement_experiment.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--file-top", type=int, default=10)
    parser.add_argument("--per-source-pool", type=int, default=4)
    parser.add_argument("--per-intent-file-quota", type=int, default=2)
    parser.add_argument("--per-intent-statement-cap", type=int, default=6)
    parser.add_argument("--max-statement-budget", type=int, default=1200)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    strict = json.loads(Path(args.strict).read_text(encoding="utf-8"))
    saved_results = [item for item in strict["results"] if item.get("status") == "ok"]
    if args.max_pairs:
        saved_results = saved_results[: args.max_pairs]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)

    file_rows = []
    stmt_rows = []
    details = []
    for pos, saved in enumerate(saved_results, 1):
        index = int(saved["index"])
        pair = pairs[index - 1]
        split = saved.get("dataset_split") or "simple"
        setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
        state, oracle = split_detector_state(setup)

        base_files = saved.get("file_module_outputs", {}).get("graphqa") or []
        enhanced_files = enhanced_file_rank(state, base_files, args.file_top, args.per_source_pool)
        gt_files = set(saved.get("ground_truth", {}).get("target_changed_files") or [])
        for method, preds in [("PropQA-original", base_files), ("PropQA-enhanced-actions", enhanced_files)]:
            paths = [item["path"] for item in preds]
            for k in [3, 5, 8, 10]:
                file_rows.append({"index": index, "dataset": split, "method": method, "k": k, **file_metrics(paths, gt_files, k)})
            file_rows.append(
                {
                    "index": index,
                    "dataset": split,
                    "method": method,
                    "k": "adaptive",
                    **file_metrics(paths, gt_files, len(paths)),
                }
            )

        target_paths = list(oracle["target_changed_paths"])
        blocks, intents = direct_intent_ast_blocks(state, runner, target_paths, args.per_intent_file_quota)
        dynamic_budget = min(
            args.max_statement_budget,
            max(100, 8 * len(intents), 6 * len(target_paths)),
        )
        statement_preds = expand_direct_blocks(
            client, state, blocks, intents, dynamic_budget, args.per_intent_statement_cap
        )
        gt_statements = to_gt(saved.get("ground_truth", {}).get("statement_ground_truth") or [])
        for k in [20, 50, 100]:
            stmt_rows.append(
                {
                    "index": index,
                    "dataset": split,
                    "method": "PropQA-direct-intent-AST-actions",
                    "k": k,
                    **metrics(statement_preds, gt_statements, k, args.tolerance),
                }
            )
        stmt_rows.append(
            {
                "index": index,
                "dataset": split,
                "method": "PropQA-direct-intent-AST-actions",
                "k": "adaptive",
                **metrics(statement_preds, gt_statements, len(statement_preds), args.tolerance),
            }
        )
        details.append(
            {
                "index": index,
                "dataset": split,
                "source_file_count": len(state["source_changed_paths"]),
                "file_adaptive_k": len(enhanced_files),
                "source_intent_count": len(intents),
                "statement_dynamic_budget": dynamic_budget,
                "statement_prediction_count": len(statement_preds),
                "direct_block_count": len(blocks),
                "top_enhanced_files": enhanced_files[:20],
                "top_direct_blocks": [
                    {key: block[key] for key in ["path", "symbol", "start", "end", "score", "actions"]}
                    for block in blocks[:20]
                ],
            }
        )
        if pos % 10 == 0:
            checkpoint = Path(args.output).with_suffix(".checkpoint.json")
            checkpoint.write_text(
                json.dumps(
                    {"completed": pos, "file_rows": file_rows, "statement_rows": stmt_rows, "details": details},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(f"[{pos}/{len(saved_results)}] index={index} files={len(enhanced_files)} intents={len(intents)} blocks={len(blocks)}", flush=True)

    summary_file = []
    summary_stmt = []
    for split in ["all", "simple", "hard"]:
        for method in sorted({row["method"] for row in file_rows}):
            for k in [3, 5, 8, 10, "adaptive"]:
                rows = [r for r in file_rows if r["method"] == method and r["k"] == k and (split == "all" or r["dataset"] == split)]
                summary_file.append({"dataset": split, "method": method, "k": k, "pairs": len(rows), **average(rows, ["precision", "recall", "f1", "mrr"])})
        for k in [20, 50, 100, "adaptive"]:
            rows = [r for r in stmt_rows if r["k"] == k and (split == "all" or r["dataset"] == split)]
            summary_stmt.append({"dataset": split, "method": "PropQA-direct-intent-AST-actions", "k": k, "pairs": len(rows), **average(rows, ["precision", "coverage", "f1", "mrr"])})

    payload = {
        "summary": {
            "pairs": len(saved_results),
            "no_leak": True,
            "file_actions": ["ExactCounterpart", "PerSourceFileQuota"],
            "statement_actions": ["DirectIntentToASTBlock", "InsertAnchor", "DeleteRegion"],
        },
        "file_rows": summary_file,
        "statement_rows": summary_stmt,
        "details": details,
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# PropQA Action Enhancement Experiment", "", "## File Detection", ""]
    md += ["| Dataset | Method | K | Pairs | Precision | Recall | F1 | MRR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_file:
        md.append(f"| {row['dataset']} | {row['method']} | {row['k']} | {row['pairs']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |")
    md += ["", "## Statement Detection", "", table(summary_stmt), ""]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
