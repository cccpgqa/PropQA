#!/usr/bin/env python3
"""Run the proposed best QA method on manually curated new hard samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_detection_tables import collect_rows, markdown_table
from propagation_detector import GitHubClient
from run_contextual_qa_judge import build_prediction, candidate_files, candidate_statements, llm_judge
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


def patch_intent(state: dict[str, Any]) -> dict[str, bool]:
    text = (
        state["propagator"].title
        + "\n"
        + state["propagator"].body
        + "\n"
        + state["propagator"].message
        + "\n"
        + source_patch_text(state)
    ).lower()
    deleted = sum(1 for stmt in state.get("source_statements", []) if stmt.kind == "-")
    return {
        "test": any("_test." in p or "/test" in p.lower() for p in state["source_changed_paths"])
        or "test" in text
        or "regression" in text,
        "symbol": any(word in text for word in ["func ", "type ", "var ", "const ", "signature", "rename", "refactor"]),
        "deletion_interface_build": deleted >= 3
        or any(word in text for word in ["delete", "remove", "protocol", "handler", "flag", "rpc", "api", "json", "marshal", "import"]),
    }


def as_agent_result(method: str, files: list[dict[str, Any]], statements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "prediction": {
            "strategy": method,
            "target_files": [{"path": f["path"], "confidence": float(f.get("score", f.get("confidence", 0.0)) or 0.0)} for f in files[:8]],
            "target_statements": statements[:8],
        },
    }


def routed_qa_candidates(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    embedder: Embedder,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    agents = []

    nicad_files, nicad_statements = baseline_open_nicad(state, runner, args)
    agents.append(as_agent_result("AskOpenNiCadEvidence", nicad_files, nicad_statements))

    c2v_files, c2v_statements = baseline_code2vec(state, runner, args)
    agents.append(as_agent_result("AskCode2VecEvidence", c2v_files, c2v_statements))

    cb_files, cb_statements = baseline_codebert(state, runner, args, embedder)
    agents.append(as_agent_result("AskCodeBertEvidence", cb_files, cb_statements))

    graph_files = action_graphqa(state, args.candidate_files)
    intents = patch_intent(state)
    if intents["test"]:
        graph_files = action_test_impact(state, graph_files, args.expanded_files)
    if intents["symbol"]:
        graph_files = action_symbol_impact(state, graph_files, args.expanded_files)
    if intents["deletion_interface_build"]:
        graph_files = action_deletion_interface_build_impact(state, graph_files, args.expanded_files)
    from run_agent_strategy_matrix import map_candidates, round_robin_statements

    graph_statements = map_candidates(runner, state, graph_files[: args.inspect_files], args.candidate_statements)
    graph_statements = round_robin_statements(graph_statements, args.top_statements)
    agents.append(as_agent_result("AskRoutedImpactPropQA", graph_files, graph_statements))
    return agents


def fallback_vote_prediction(agent_results: list[dict[str, Any]], top_files: int, top_statements: int) -> dict[str, Any]:
    file_scores: dict[str, float] = {}
    stmt_scores: dict[tuple[str, int], dict[str, Any]] = {}
    for agent_weight, result in enumerate(agent_results, start=1):
        pred = result["prediction"]
        weight = 1.0 + 0.04 * agent_weight
        for rank, item in enumerate(pred.get("target_files", []), start=1):
            path = item.get("path")
            if not path:
                continue
            score = weight * (1.0 / rank + float(item.get("confidence") or 0.0))
            file_scores[path] = file_scores.get(path, 0.0) + score
        for rank, item in enumerate(pred.get("target_statements", []), start=1):
            path = item.get("path")
            try:
                line = int(item.get("line"))
            except Exception:
                continue
            key = (path, line)
            score = weight * (1.0 / rank + float(item.get("score") or item.get("confidence") or 0.0))
            old = stmt_scores.get(key)
            if old is None:
                old = dict(item)
                old["_vote_score"] = 0.0
                stmt_scores[key] = old
            old["_vote_score"] += score
    files = [
        {"path": path, "confidence": round(score, 6)}
        for path, score in sorted(file_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_files]
    ]
    statements = sorted(stmt_scores.values(), key=lambda item: (-float(item.get("_vote_score", 0.0)), item["path"], int(item["line"])))[:top_statements]
    return {"propagation_likely": bool(files), "target_files": files, "target_statements": statements, "strategy": "routed_qa_vote", "llm_used": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), os.environ.get("GITHUB_TOKEN"), sleep_seconds=args.sleep)
    llm = LoopLLM(timeout=args.llm_timeout)
    embedder = Embedder(args.embedding_model)
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results_vote = []
    results_llm = []
    skipped = []
    for index, pair in enumerate(pairs, start=1):
        runner = PropagationQAMCTS(client, max_nodes=8, top_files=args.top_files, top_statements=args.top_statements)
        state = prepare_state(runner, pair)
        if state.get("status") and state.get("status") != "ok":
            skipped.append({"index": index, "reason": state.get("reason")})
            continue
        source_summary = runner._execute(state, {"type": "InspectSourcePatch"})
        runner._update_state(state, {"type": "InspectSourcePatch"}, source_summary)

        agent_results = routed_qa_candidates(state, runner, embedder, args)
        vote_prediction = fallback_vote_prediction(agent_results, args.top_files, args.top_statements)
        vote_metrics = evaluate_prediction(state, vote_prediction)
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
            "debug": {"intent": patch_intent(state), "agents": [r["method"] for r in agent_results]},
        }
        results_vote.append({**common, "method": "Routed-QA-Vote", "prediction": vote_prediction, "metrics": vote_metrics})

        files = candidate_files(*agent_results)
        statements = candidate_statements(*agent_results)
        llm_json = None
        llm_error = ""
        if llm.enabled:
            llm_json, llm_error = llm_judge(llm, pair, state, files, statements, hide_target_metadata=args.hide_target_metadata)
        llm_prediction = build_prediction(llm_json, files, statements, args.top_files, args.top_statements)
        llm_prediction["strategy"] = "routed_qa_llm_judge"
        llm_metrics = evaluate_prediction(state, llm_prediction)
        results_llm.append(
            {
                **common,
                "method": "Routed-QA-LLM",
                "prediction": llm_prediction,
                "metrics": llm_metrics,
                "llm_error": llm_error,
            }
        )
    rows = []
    hard_set = set(range(1, len(pairs) + 1))
    rows.extend(collect_rows("new_hard_best_qa", "Routed-QA-Vote", results_vote, hard_set, args.tolerance, datasets=("all",)))
    rows.extend(collect_rows("new_hard_best_qa", "Routed-QA-LLM", results_llm, hard_set, args.tolerance, datasets=("all",)))
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
        "results": {"Routed-QA-Vote": results_vote, "Routed-QA-LLM": results_llm},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/manual_curated_new_hard_samples.json")
    parser.add_argument("--output", default="outputs/new_hard_best_qa_results.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--embedding-model", default="D:/Python/Blockchain/sentenceTransformer/all-MiniLM-L12-v2")
    parser.add_argument("--candidate-files", type=int, default=24)
    parser.add_argument("--expanded-files", type=int, default=48)
    parser.add_argument("--prefilter-files", type=int, default=80)
    parser.add_argument("--inspect-files", type=int, default=16)
    parser.add_argument("--candidate-statements", type=int, default=32)
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
    md = out.with_suffix(".md")
    md.write_text(
        "# Best QA on Hard Samples\n\n"
        + f"- Samples: {payload['summary']['sample_count']}\n"
        + f"- Skipped: {len(payload['summary']['skipped'])}\n"
        + f"- LLM enabled: {payload['summary']['llm_enabled']}\n\n"
        + markdown_table("Best QA Results", payload["rows"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
