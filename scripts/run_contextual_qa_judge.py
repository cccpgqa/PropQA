#!/usr/bin/env python3
"""Contextual QA judge for propagation localization candidates.

This script keeps the task in QA form: given propagator metadata
(title/body/message), target metadata, source patch snippets, and candidate
answers from multiple graph/code agents, the LLM selects the best file and
statement answers. It does not invent paths or lines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from propagation_detector import GitHubClient, normalize_repo, parsed_sides, resolve_side
from run_agent_strategy_matrix import DEFAULT_HARD, safe_ints
from run_llm_in_loop_hard_cases import LoopLLM, evaluate_prediction
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv


def clip(text: str, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def raw_text_context(raw: dict[str, Any]) -> dict[str, Any]:
    resolved = raw.get("resolved_by") if isinstance(raw.get("resolved_by"), dict) else {}
    return {
        "project": raw.get("project"),
        "repo": normalize_repo(raw.get("project") or ""),
        "type": raw.get("type"),
        "number": raw.get("pr_number") or raw.get("issue_number") or raw.get("number"),
        "title": clip(raw.get("title") or resolved.get("title") or raw.get("message") or resolved.get("message") or "", 320),
        "body": clip(raw.get("body") or resolved.get("body") or "", 1000),
        "message": clip(raw.get("message") or resolved.get("message") or raw.get("commit_message") or resolved.get("commit_message") or "", 650),
        "url": raw.get("html_url") or resolved.get("html_url"),
        "file_names": list(resolved.get("file_names") or raw.get("file_names") or [])[:25],
    }


def ordered_metadata(pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_raw = raw_text_context(pair.get("Source", {}))
    target_raw = raw_text_context(pair.get("Infestor", {}))
    return source_raw, target_raw


def setup_state(pair: dict[str, Any], client: GitHubClient) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=8, top_statements=8)
    return runner._prepare_pair(pair)


def source_patch_context(setup: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": stmt.path,
            "line": stmt.line_no,
            "kind": stmt.kind,
            "text": clip(stmt.text, 180),
        }
        for stmt in setup.get("source_statements", [])[:24]
    ]


def candidate_files(*results: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for result in results:
        for rank, item in enumerate((result.get("prediction") or {}).get("target_files", []), start=1):
            path = item.get("path")
            if not path or path in seen:
                continue
            seen.add(path)
            out.append(
                {
                    "path": path,
                    "rank": rank,
                    "confidence": item.get("confidence", item.get("score")),
                    "agent": (result.get("prediction") or {}).get("strategy") or result.get("method") or "agent",
                }
            )
    return out[:16]


def candidate_statements(*results: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for result in results:
        agent = (result.get("prediction") or {}).get("strategy") or result.get("method") or "agent"
        for rank, item in enumerate((result.get("prediction") or {}).get("target_statements", []), start=1):
            path = item.get("path")
            try:
                line = int(item.get("line"))
            except (TypeError, ValueError):
                continue
            key = (path, line)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "path": path,
                    "line": line,
                    "rank": rank,
                    "agent": agent,
                    "score": item.get("score", item.get("confidence")),
                    "line_similarity": item.get("line_similarity"),
                    "symbol_score": item.get("symbol_score"),
                    "pattern_score": item.get("pattern_score"),
                    "patch_pattern": item.get("patch_pattern"),
                    "target_symbol": item.get("target_symbol"),
                    "source_path": item.get("source_path") or item.get("matched_source_path"),
                    "source_text": clip(item.get("source_text") or "", 180),
                    "target_text": clip(item.get("line_text") or "", 180),
                }
            )
    return out[:32]


def llm_judge(
    llm: LoopLLM,
    pair: dict[str, Any],
    setup: dict[str, Any],
    file_candidates: list[dict[str, Any]],
    statement_candidates: list[dict[str, Any]],
    hide_target_metadata: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    source_meta, target_meta = ordered_metadata(pair)
    if hide_target_metadata:
        target_meta = {
            "project": target_meta.get("project"),
            "repo": target_meta.get("repo"),
            "type": "unknown_target_change",
            "number": None,
            "title": "",
            "body": "",
            "message": "",
            "url": None,
            "file_names": [],
            "available_information": (
                "Only the target repository pre-fix tree/code and retrieved candidate files/statements "
                "are available. The target PR/commit title, body, message, and changed files are hidden "
                "to avoid oracle leakage."
            ),
        }
    payload = {
        "question": (
            "A change from the propagator repository may have propagated to the target repository. "
            "Using metadata, source patch intent, and graph/code candidates, answer which target files "
            "and target pre-fix statements are most likely the propagation locations."
        ),
        "propagator_metadata": source_meta,
        "target_metadata": target_meta,
        "source_patch": {
            "changed_files": setup.get("source_changed_paths", [])[:25],
            "statements": source_patch_context(setup),
        },
        "candidate_files": file_candidates,
        "candidate_statements": statement_candidates,
        "output_schema": {
            "target_files": [{"path": "candidate path", "confidence": 0.0}],
            "target_statements": [{"path": "candidate path", "line": 1, "confidence": 0.0}],
        },
        "constraints": [
            "Return strict JSON only.",
            "Do not invent paths or line numbers.",
            "Every selected file must appear in candidate_files.",
            "Every selected statement path+line must appear in candidate_statements.",
            "Prioritize semantic consistency with title/body/message and source patch intent.",
            "Use code evidence as a guard: AST/symbol, patch pattern, exact/near statement text, and agent agreement.",
        ],
    }
    system = (
        "You are a QA evaluator in a multi-agent code propagation detector. "
        "You synthesize repository metadata and graph/code evidence, but your answer must be selected from candidates only."
    )
    result = llm.complete_json_with_system(system, payload)
    return result, llm.last_error


def build_prediction(
    llm_json: dict[str, Any] | None,
    files: list[dict[str, Any]],
    statements: list[dict[str, Any]],
    top_files: int,
    top_statements: int,
) -> dict[str, Any]:
    file_by_path = {item["path"]: item for item in files}
    stmt_by_key = {(item["path"], int(item["line"])): item for item in statements}
    selected_files = []
    selected_statements = []
    if llm_json:
        for item in llm_json.get("target_files", []):
            path = item.get("path") if isinstance(item, dict) else None
            if path in file_by_path and path not in {x["path"] for x in selected_files}:
                selected_files.append({"path": path, "confidence": float(item.get("confidence", 0.0) or 0.0), "qa_agent": "contextual_llm"})
        for item in llm_json.get("target_statements", []):
            path = item.get("path") if isinstance(item, dict) else None
            try:
                line = int(item.get("line"))
            except (TypeError, ValueError):
                continue
            if (path, line) in stmt_by_key and (path, line) not in {(x["path"], int(x["line"])) for x in selected_statements}:
                old = dict(stmt_by_key[(path, line)])
                old["llm_confidence"] = float(item.get("confidence", 0.0) or 0.0)
                old["qa_agent"] = "contextual_llm"
                selected_statements.append(old)
    for item in files:
        if len(selected_files) >= top_files:
            break
        if item["path"] not in {x["path"] for x in selected_files}:
            selected_files.append({"path": item["path"], "confidence": float(item.get("confidence") or 0.0)})
    for item in statements:
        if len(selected_statements) >= top_statements:
            break
        if (item["path"], int(item["line"])) not in {(x["path"], int(x["line"])) for x in selected_statements}:
            selected_statements.append(item)
    return {
        "propagation_likely": bool(selected_files),
        "target_files": selected_files[:top_files],
        "target_statements": selected_statements[:top_statements],
        "strategy": "contextual_qa_llm_judge",
        "llm_used": bool(llm_json),
    }


def run_one(
    index: int,
    pair: dict[str, Any],
    setup: dict[str, Any],
    llm: LoopLLM,
    agent_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    files = candidate_files(*agent_results)
    statements = candidate_statements(*agent_results)
    llm_json, llm_error = llm_judge(llm, pair, setup, files, statements, hide_target_metadata=args.hide_target_metadata)
    prediction = build_prediction(llm_json, files, statements, args.top_files, args.top_statements)
    prediction["llm_error"] = None if llm_json else llm_error
    metrics = evaluate_prediction(setup, prediction)
    return {
        "index": index,
        "status": "ok",
        "prediction": prediction,
        "metrics": metrics,
        "llm_judge": llm_json,
        "ground_truth": {
            "target_changed_files": setup["target_changed_paths"],
            "target_changed_old_line_ranges": {
                path: [{"start": start, "end": end} for start, end in ranges]
                for path, ranges in setup["target_ranges"].items()
            },
        },
        "debug": {
            "candidate_file_count": len(files),
            "candidate_statement_count": len(statements),
        },
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
        "llm_used": sum(1 for r in ok if r.get("prediction", {}).get("llm_used")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/URL_Results_detection_subset.json")
    parser.add_argument("--graphqa", default="outputs/qa_mcts_ast_detection_subset_top8.json")
    parser.add_argument("--final", default="outputs/final_llm_hybrid_detection_subset_top8.json")
    parser.add_argument("--patch", default="outputs/patch_tracing_baseline_detection_subset_top8.json")
    parser.add_argument("--indices", default=DEFAULT_HARD)
    parser.add_argument("--output", default="outputs/contextual_qa_judge_hard15.json")
    parser.add_argument("--top-files", type=int, default=8)
    parser.add_argument("--top-statements", type=int, default=8)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--hide-target-metadata", action="store_true", default=True)
    parser.add_argument("--allow-target-metadata", dest="hide_target_metadata", action="store_false")
    args = parser.parse_args()

    load_dotenv()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    indices = safe_ints(args.indices)
    sources = []
    for path in [args.graphqa, args.final, args.patch]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources.append({int(r["index"]): r for r in data.get("results", []) if r.get("status") == "ok"})
    client = GitHubClient(Path(".cache/github"), token=os.environ.get("GITHUB_TOKEN"))
    llm = LoopLLM(timeout=args.llm_timeout)
    results = []
    for offset, index in enumerate(indices, start=1):
        try:
            pair = pairs[index - 1]
            setup = setup_state(pair, client)
            if setup.get("status") != "ok":
                result = {"index": index, **setup}
            else:
                agent_results = [source[index] for source in sources if index in source]
                result = run_one(index, pair, setup, llm, agent_results, args)
        except Exception as exc:
            result = {"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        Path(args.output).write_text(json.dumps({"summary": summarize(results), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{offset}/{len(indices)} index={index}] {result['status']} {result.get('reason','')}", flush=True)
    payload = {"summary": summarize(results), "results": results}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
