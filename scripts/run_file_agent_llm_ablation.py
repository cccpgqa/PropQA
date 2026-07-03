#!/usr/bin/env python3
"""Test whether an LLM action agent improves file-stage ranking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_detection_tables import collect_rows, markdown_table
from propagation_detector import GitHubClient
from run_agent_strategy_matrix import map_candidates, round_robin_statements
from run_best_qa_new_hard import patch_intent
from run_contextual_qa_judge import build_prediction, candidate_files, candidate_statements, clip, llm_judge
from run_llm_in_loop_hard_cases import LoopLLM, evaluate_prediction, prepare_state
from run_new_hard_baseline_ablation import (
    Embedder,
    action_deletion_interface_build_impact,
    action_graphqa,
    action_symbol_impact,
    action_test_impact,
    baseline_code2vec,
    baseline_codebert,
    baseline_open_nicad,
    source_patch_text,
)
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv


def as_agent_result(method: str, files: list[dict[str, Any]], statements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "prediction": {
            "strategy": method,
            "target_files": [{"path": f["path"], "confidence": float(f.get("score", f.get("confidence", 0.0)) or 0.0)} for f in files[:8]],
            "target_statements": statements[:8],
        },
    }


def graph_impact_files(state: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    files = action_graphqa(state, args.candidate_files)
    intents = patch_intent(state)
    if intents["test"]:
        files = action_test_impact(state, files, args.expanded_files)
    if intents["symbol"]:
        files = action_symbol_impact(state, files, args.expanded_files)
    if intents["deletion_interface_build"]:
        files = action_deletion_interface_build_impact(state, files, args.expanded_files)
    return files


def collect_file_action_candidates(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    embedder: Embedder,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    graph_files = graph_impact_files(state, args)
    nicad_files, nicad_statements = baseline_open_nicad(state, runner, args)
    code2vec_files, code2vec_statements = baseline_code2vec(state, runner, args)
    codebert_files, codebert_statements = baseline_codebert(state, runner, args, embedder)
    action_outputs = {
        "FindFile": graph_files[:12],
        "FindSymbol": code2vec_files[:12],
        "FindCodeSnippet": nicad_files[:12],
        "SemanticSearch": codebert_files[:12],
    }
    merged = {}
    for action, files in action_outputs.items():
        for rank, item in enumerate(files, start=1):
            path = item.get("path")
            if not path:
                continue
            old = merged.setdefault(
                path,
                {
                    "path": path,
                    "score": 0.0,
                    "matched_source_path": item.get("matched_source_path"),
                    "actions": [],
                    "evidence": {},
                },
            )
            score = float(item.get("score", item.get("confidence", 0.0)) or 0.0)
            old["score"] += score + 1.0 / rank
            old["actions"].append(action)
            old["evidence"][action] = round(score, 6)
            if not old.get("matched_source_path"):
                old["matched_source_path"] = item.get("matched_source_path")
    candidates = sorted(merged.values(), key=lambda x: (-x["score"], x["path"]))[: args.file_agent_candidates]
    support_statements = nicad_statements[:8] + code2vec_statements[:8] + codebert_statements[:8]
    return candidates, action_outputs, support_statements


def file_agent_llm_rank(
    llm: LoopLLM,
    pair: dict[str, Any],
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    action_outputs: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    if not llm.enabled:
        return candidates[: args.candidate_files], "LLM disabled"
    source = pair.get("Source", {})
    target = pair.get("Infestor", {})
    payload = {
        "question": "Choose target files most likely affected by propagation, using action evidence only.",
        "source_metadata": {
            "title": clip(source.get("title") or source.get("message") or "", 400),
            "body": clip(source.get("body") or "", 1000),
            "message": clip(source.get("message") or "", 600),
            "files": state.get("source_changed_paths", [])[:25],
        },
        "target_metadata": {
            "project": target.get("project"),
            "repo": target.get("repo"),
            "available_information": (
                "Target PR/commit title, body, message, and changed files are hidden. "
                "Use only target pre-fix repository candidates returned by actions."
            ),
        },
        "source_patch_excerpt": clip(source_patch_text(state), 1800),
        "action_space": {
            "FindFile": "name/path/tree retrieval",
            "FindSymbol": "symbol/path-context retrieval",
            "FindCodeSnippet": "normalized clone retrieval",
            "SemanticSearch": "semantic code retrieval",
            "Finish": "select final files from candidates",
        },
        "action_observations": {
            name: [
                {
                    "path": item.get("path"),
                    "rank": rank,
                    "score": item.get("score", item.get("confidence")),
                    "matched_source_path": item.get("matched_source_path"),
                }
                for rank, item in enumerate(items[:10], start=1)
            ]
            for name, items in action_outputs.items()
        },
        "candidate_files": [
            {"path": item["path"], "actions": item.get("actions", []), "evidence": item.get("evidence", {})}
            for item in candidates
        ],
        "output_schema": {"target_files": [{"path": "candidate path", "confidence": 0.0, "reason": "short"}]},
        "constraints": [
            "Return strict JSON only.",
            "Select only paths from candidate_files.",
            "Do not invent paths.",
            "Prefer files supported by multiple actions and by source/target metadata intent.",
        ],
    }
    system = "You are the file-stage QA agent. You plan over retrieval action results and finish with ranked candidate files."
    result = llm.complete_json_with_system(system, payload)
    if not result:
        return candidates[: args.candidate_files], llm.last_error or "no LLM JSON"
    by_path = {item["path"]: item for item in candidates}
    selected = []
    seen = set()
    for item in result.get("target_files", []):
        path = item.get("path") if isinstance(item, dict) else None
        if path in by_path and path not in seen:
            old = dict(by_path[path])
            old["llm_confidence"] = float(item.get("confidence", 0.0) or 0.0)
            old["llm_reason"] = item.get("reason", "")
            old["score"] = old["score"] + 2.0 * old["llm_confidence"]
            selected.append(old)
            seen.add(path)
    for item in candidates:
        if len(selected) >= args.candidate_files:
            break
        if item["path"] not in seen:
            selected.append(item)
            seen.add(item["path"])
    return selected[: args.candidate_files], ""


def run_method(
    method: str,
    pair: dict[str, Any],
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    embedder: Embedder,
    llm: LoopLLM,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates, action_outputs, support_statements = collect_file_action_candidates(state, runner, embedder, args)
    if method == "FileAgent-LLM":
        file_candidates, file_error = file_agent_llm_rank(llm, pair, state, candidates, action_outputs, args)
    elif method == "FileAgent-NoLLM":
        file_candidates, file_error = candidates[: args.candidate_files], ""
    else:
        raise ValueError(method)

    mapped = map_candidates(runner, state, file_candidates[: args.inspect_files], args.candidate_statements)
    mapped = round_robin_statements(mapped, args.top_statements)
    agent_results = [
        as_agent_result(method, file_candidates, mapped),
        as_agent_result("StatementSupport", file_candidates, support_statements),
    ]
    files = candidate_files(*agent_results)
    statements = candidate_statements(*agent_results)
    llm_json = None
    judge_error = ""
    if llm.enabled:
        llm_json, judge_error = llm_judge(llm, pair, state, files, statements, hide_target_metadata=args.hide_target_metadata)
    prediction = build_prediction(llm_json, files, statements, args.top_files, args.top_statements)
    prediction["strategy"] = method
    prediction["file_agent_error"] = file_error
    prediction["judge_error"] = judge_error
    return prediction


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), os.environ.get("GITHUB_TOKEN"), sleep_seconds=args.sleep)
    llm = LoopLLM(timeout=args.llm_timeout)
    embedder = Embedder(args.embedding_model)
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    methods = {"FileAgent-NoLLM": [], "FileAgent-LLM": []}
    skipped = []
    for index, pair in enumerate(pairs, start=1):
        runner = PropagationQAMCTS(client, max_nodes=8, top_files=args.top_files, top_statements=args.top_statements)
        state = prepare_state(runner, pair)
        if state.get("status") and state.get("status") != "ok":
            skipped.append({"index": index, "reason": state.get("reason")})
            continue
        source_summary = runner._execute(state, {"type": "InspectSourcePatch"})
        runner._update_state(state, {"type": "InspectSourcePatch"}, source_summary)
        common = {
            "index": index,
            "status": "ok",
            "ground_truth": {
                "target_changed_files": state["target_changed_paths"],
                "target_changed_old_line_ranges": {
                    path: [{"start": start, "end": end} for start, end in ranges]
                    for path, ranges in state["target_ranges"].items()
                },
            },
        }
        for method in methods:
            prediction = run_method(method, pair, state, runner, embedder, llm, args)
            metrics = evaluate_prediction(state, prediction)
            methods[method].append({**common, "method": method, "prediction": prediction, "metrics": metrics})
    rows = []
    hard = set(range(1, len(pairs) + 1))
    for method, results in methods.items():
        rows.extend(collect_rows("file_agent_llm_ablation", method, results, hard, args.tolerance, datasets=("all",)))
    return {
        "summary": {
            "input": args.input,
            "sample_count": len(pairs),
            "skipped": skipped,
            "llm_enabled": llm.enabled,
            "embedding_model": args.embedding_model,
            "hide_target_metadata": args.hide_target_metadata,
        },
        "rows": rows,
        "results": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/combined_hard_samples.json")
    parser.add_argument("--output", default="outputs/file_agent_llm_ablation.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--embedding-model", default="D:/Python/Blockchain/sentenceTransformer/all-MiniLM-L12-v2")
    parser.add_argument("--candidate-files", type=int, default=28)
    parser.add_argument("--file-agent-candidates", type=int, default=36)
    parser.add_argument("--expanded-files", type=int, default=56)
    parser.add_argument("--prefilter-files", type=int, default=80)
    parser.add_argument("--inspect-files", type=int, default=18)
    parser.add_argument("--candidate-statements", type=int, default=40)
    parser.add_argument("--top-files", type=int, default=8)
    parser.add_argument("--top-statements", type=int, default=8)
    parser.add_argument("--max-file-chars", type=int, default=12000)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--hide-target-metadata", action="store_true", default=True)
    parser.add_argument("--allow-target-metadata", dest="hide_target_metadata", action="store_false")
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(
        "# File-Stage LLM Agent Ablation\n\n"
        + f"- Samples: {payload['summary']['sample_count']}\n"
        + f"- Skipped: {len(payload['summary']['skipped'])}\n"
        + f"- LLM enabled: {payload['summary']['llm_enabled']}\n\n"
        + markdown_table("File Agent Results", payload["rows"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
