#!/usr/bin/env python3
"""Evaluate several hard-case agent strategies.

Strategies:
- recall_first: inspect all high-recall file candidates before final ranking.
- neighbor_expand: recall_first plus same-directory/test-file neighbor expansion.
- intent_round_robin: preserve per-source-file patch intents when ranking statements.
- evidence_vote: combine file, statement, and simple token/path evidence for final ranking.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_detection_tables import collect_rows
from propagation_detector import GitHubClient, jaccard, path_score, path_tokens, token_similarity
from run_llm_in_loop_hard_cases import DEFAULT_HARD, evaluate_prediction, prepare_state
from run_qa_mcts_small_sample import PropagationQAMCTS, dedupe_statement_predictions, load_dotenv


def safe_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip().isdigit()]


def norm_path(path: str) -> str:
    return path.lower().replace("_", "").replace("-", "").replace("/", "")


def source_terms(state: dict[str, Any]) -> set[str]:
    terms = set()
    for path in state["source_changed_paths"]:
        terms |= path_tokens(path)
    for stmt in state["source_statements"][:80]:
        terms |= {tok.lower() for tok in stmt.text.replace('"', " ").replace("'", " ").split() if len(tok) >= 4}
    return {t.strip(".,:;()[]{}") for t in terms if len(t.strip(".,:;()[]{}")) >= 4}


def rank_files_high_recall(runner: PropagationQAMCTS, state: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    terms = source_terms(state)
    for source_path in state["source_changed_paths"]:
        src_norm = norm_path(source_path)
        src_tokens = path_tokens(source_path)
        for target_path in state["target_tree"]:
            base = path_score(source_path, target_path)
            normalized = 1.0 if src_norm == norm_path(target_path) else 0.0
            token = jaccard(src_tokens | terms, path_tokens(target_path))
            score = 0.72 * base + 0.18 * normalized + 0.10 * token
            old = candidates.get(target_path)
            if old is None or score > old["score"]:
                candidates[target_path] = {
                    "path": target_path,
                    "score": round(score, 6),
                    "matched_source_path": source_path,
                    "evidence": {"path_score": round(base, 6), "normalized_match": normalized, "token_score": round(token, 6)},
                }
    return sorted(candidates.values(), key=lambda item: (-item["score"], item["path"]))[:top_k]


def neighbor_expand(state: dict[str, Any], candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_path = {item["path"]: dict(item) for item in candidates}
    source_names = {p.rsplit("/", 1)[-1].replace("_", "").replace("-", "").lower() for p in state["source_changed_paths"]}
    tree = state["target_tree"]
    for item in candidates[:20]:
        path = item["path"]
        directory = path.rsplit("/", 1)[0] if "/" in path else ""
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        for target in tree:
            if target in by_path:
                continue
            if directory and not target.startswith(directory + "/"):
                continue
            name = target.rsplit("/", 1)[-1]
            name_norm = name.replace("_", "").replace("-", "").lower()
            is_test_pair = (stem + "_test") in name or stem.replace("_test", "") in name
            source_like = name_norm in source_names
            same_ext = bool(ext and target.endswith("." + ext))
            if same_ext and (is_test_pair or source_like or token_similarity(name, path.rsplit("/", 1)[-1]) > 0.55):
                by_path[target] = {
                    "path": target,
                    "score": round(float(item["score"]) * 0.92, 6),
                    "matched_source_path": item.get("matched_source_path"),
                    "evidence": {"neighbor_of": path},
                }
    return sorted(by_path.values(), key=lambda item: (-float(item["score"]), item["path"]))[:limit]


def map_candidates(
    runner: PropagationQAMCTS,
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    candidate_statements: int,
) -> list[dict[str, Any]]:
    old_top = runner.top_statements
    runner.top_statements = candidate_statements
    all_items = []
    for item in candidates:
        mapped = runner._map_patch_to_target(state, item["path"], item.get("matched_source_path"))
        for stmt in mapped:
            stmt = dict(stmt)
            stmt["file_candidate_score"] = item.get("score", 0.0)
            stmt["matched_source_path"] = item.get("matched_source_path")
            all_items.append(stmt)
    runner.top_statements = old_top
    all_items.sort(key=lambda x: (-float(x.get("score", 0.0)), -float(x.get("file_candidate_score", 0.0)), x["path"], int(x["line"])))
    return dedupe_statement_predictions(all_items)


def rank_files_from_evidence(candidates: list[dict[str, Any]], statements: list[dict[str, Any]], top_files: int, evidence_vote: bool) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for rank, item in enumerate(candidates):
        base = float(item.get("score", 0.0))
        scores[item["path"]] = max(scores.get(item["path"], 0.0), 0.55 * base + 0.02 * max(0, 20 - rank))
    for item in statements:
        stmt_score = float(item.get("score", 0.0))
        file_score = float(item.get("file_candidate_score", 0.0))
        if evidence_vote:
            combined = 0.52 * stmt_score + 0.35 * file_score + 0.13 * float(item.get("symbol_score", 0.0))
        else:
            combined = stmt_score
        scores[item["path"]] = max(scores.get(item["path"], 0.0), combined)
    return [
        {"path": path, "confidence": round(score, 6)}
        for path, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_files]
    ]


def round_robin_statements(statements: list[dict[str, Any]], top_statements: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in statements:
        key = item.get("matched_source_path") or item.get("source_path") or item["path"]
        by_source[key].append(item)
    for items in by_source.values():
        items.sort(key=lambda x: (-float(x.get("score", 0.0)), x["path"], int(x["line"])))
    out = []
    while len(out) < top_statements and by_source:
        progressed = False
        for key in list(by_source):
            if by_source[key]:
                out.append(by_source[key].pop(0))
                progressed = True
                if len(out) >= top_statements:
                    break
            if not by_source[key]:
                by_source.pop(key, None)
        if not progressed:
            break
    return dedupe_statement_predictions(out)[:top_statements]


def run_strategy(pair: dict[str, Any], index: int, client: GitHubClient, args: argparse.Namespace, strategy: str) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=args.top_files, top_statements=args.top_statements)
    state = prepare_state(runner, pair)
    if state.get("status") and state.get("status") != "ok":
        return {"index": index, **state}
    source_summary = runner._execute(state, {"type": "InspectSourcePatch"})
    runner._update_state(state, {"type": "InspectSourcePatch"}, source_summary)

    candidates = rank_files_high_recall(runner, state, args.candidate_files)
    if strategy in {"neighbor_expand", "intent_round_robin", "evidence_vote"}:
        candidates = neighbor_expand(state, candidates, args.expanded_files)
    state["candidate_files"] = candidates

    inspect_count = args.inspect_files
    statements = map_candidates(runner, state, candidates[:inspect_count], args.candidate_statements)
    if strategy == "intent_round_robin":
        target_statements = round_robin_statements(statements, args.top_statements)
    elif strategy == "evidence_vote":
        target_statements = sorted(
            statements,
            key=lambda item: (
                -(0.52 * float(item.get("score", 0.0)) + 0.35 * float(item.get("file_candidate_score", 0.0)) + 0.13 * float(item.get("symbol_score", 0.0))),
                item["path"],
                int(item["line"]),
            ),
        )[: args.top_statements]
    else:
        target_statements = statements[: args.top_statements]
    target_files = rank_files_from_evidence(candidates, target_statements, args.top_files, strategy == "evidence_vote")
    prediction = {
        "propagation_likely": bool(target_files),
        "target_files": target_files,
        "target_statements": target_statements,
        "strategy": strategy,
    }
    metrics = evaluate_prediction(state, prediction)
    return {
        "index": index,
        "status": "ok",
        "prediction": prediction,
        "metrics": metrics,
        "ground_truth": {
            "target_changed_files": state["target_changed_paths"],
            "target_changed_old_line_ranges": {
                path: [{"start": start, "end": end} for start, end in ranges]
                for path, ranges in state["target_ranges"].items()
            },
        },
        "propagator": {
            "repo": state["propagator"].repo,
            "number": state["propagator"].number,
            "title": state["propagator"].title,
            "file_names": state["source_changed_paths"],
        },
        "target": {
            "repo": state["target"].repo,
            "number": state["target"].number,
            "title": state["target"].title,
            "file_names": state["target_changed_paths"],
        },
        "debug": {"candidate_count": len(candidates), "mapped_statement_count": len(statements)},
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "evaluated_pairs": len(ok),
        "skipped_pairs": len(results) - len(ok),
        "file_top8_hits": sum(1 for r in ok if r.get("metrics", {}).get("file_topk_hit")),
        "file_top8_hit_rate": round(sum(1 for r in ok if r.get("metrics", {}).get("file_topk_hit")) / len(ok), 4) if ok else None,
        "statement_top8_hits": sum(1 for r in ok if r.get("metrics", {}).get("statement_topk_hit")),
        "statement_top8_hit_rate": round(sum(1 for r in ok if r.get("metrics", {}).get("statement_topk_hit")) / len(ok), 4) if ok else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/URL_Results_detection_subset.json")
    parser.add_argument("--output-dir", default="outputs/agent_strategy_matrix")
    parser.add_argument("--indices", default=DEFAULT_HARD)
    parser.add_argument("--strategies", default="recall_first,neighbor_expand,intent_round_robin,evidence_vote")
    parser.add_argument("--candidate-files", type=int, default=28)
    parser.add_argument("--expanded-files", type=int, default=36)
    parser.add_argument("--inspect-files", type=int, default=28)
    parser.add_argument("--candidate-statements", type=int, default=36)
    parser.add_argument("--top-files", type=int, default=8)
    parser.add_argument("--top-statements", type=int, default=8)
    args = parser.parse_args()

    load_dotenv()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    indices = safe_ints(args.indices)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GitHubClient(Path(".cache/github"), token=os.environ.get("GITHUB_TOKEN"))
    summary_rows = []
    for strategy in strategies:
        results = []
        for offset, index in enumerate(indices, start=1):
            try:
                result = run_strategy(pairs[index - 1], index, client, args, strategy)
            except Exception as exc:
                result = {"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            payload = {"summary": summarize(results), "results": results}
            (out_dir / f"{strategy}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{strategy} {offset}/{len(indices)} index={index}] {result['status']} {result.get('reason','')}", flush=True)
        payload = {"summary": summarize(results), "results": results}
        (out_dir / f"{strategy}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.extend(collect_rows("strategy_matrix", strategy, results, set(indices), 5, datasets=("hard",)))
        print(strategy, json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    (out_dir / "strategy_metrics.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
