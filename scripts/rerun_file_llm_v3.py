#!/usr/bin/env python3
"""LLM-in-the-loop file-level PropQA evaluation for one configured LLM."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

from propagation_detector import GitHubClient
from rerun_statement_existing_code_v3 import LLMClient
from run_file_pair_detection_experiment import build_file_pair_records, restrict_state_to_source_file
from run_llm_in_loop_hard_cases import prepare_state
from run_no_leak_five_module_pipeline import normalize_target_oracle_to_dataset_files, split_detector_state
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv
from run_rq5_action_ablation_v3 import file_action_rank, file_metrics_for_records


BASELINES = ("Open-NiCad", "code2vec", "Open-NiCad+Path", "code2vec+Path")


def compact_source_statements(state: dict[str, Any], source_file: str) -> list[dict[str, Any]]:
    return [
        {"operation": statement.kind, "text": statement.text[:160]}
        for statement in state.get("source_statements", [])
        if statement.path == source_file
    ][:16]


def candidate_symbols(
    runner: PropagationQAMCTS,
    state: dict[str, Any],
    path: str,
) -> list[str]:
    if path not in set(state["target_tree"]):
        return []
    try:
        symbols = runner._symbols_for_file(
            state, "target", state["target"].repo, state["target"].base_sha, path
        )
        return [str(symbol.get("name") or "") for symbol in symbols if symbol.get("name")][:8]
    except Exception:
        return []


def candidate_code_preview(
    runner: PropagationQAMCTS,
    state: dict[str, Any],
    path: str,
    symbols: list[str],
) -> str:
    if path not in set(state["target_tree"]):
        return ""
    content = runner.client.file_at(state["target"].repo, state["target"].base_sha, path) or ""
    lines = content.splitlines()
    if not lines:
        return ""
    center = 0
    for symbol in symbols:
        for index, line in enumerate(lines):
            if symbol and symbol in line:
                center = index
                break
        if center:
            break
    start = max(0, center - 4)
    return "\n".join(lines[start : start + 16])[:1200]


def _valid_ids(values: Any, upper_bound: int) -> list[int]:
    result = []
    for value in values or []:
        try:
            candidate_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= candidate_id < upper_bound and candidate_id not in result:
            result.append(candidate_id)
    return result


def rerank_batch(
    llm: LLMClient,
    state: dict[str, Any],
    queries: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    source = state["propagator"]
    prompt_queries = []
    candidate_lookup: dict[tuple[str, int], str] = {}
    for query_index, query in enumerate(queries):
        candidates = []
        for candidate_id, item in enumerate(query["candidates"]):
            candidate_lookup[(query["source_file"], candidate_id)] = item["path"]
            candidates.append(
                {
                    "id": candidate_id,
                    "path": item["path"],
                    "action": item.get("action") or "FindCounterpart",
                    "structural_score": round(float(item.get("score", 0.0)), 6),
                    "matched_source_path": item.get("matched_source_path"),
                    "target_symbols": item.get("target_symbols", []),
                    "target_pre_fix_code": item.get("target_code_preview", ""),
                }
            )
        prompt_queries.append(
            {
                "query_id": query_index,
                "source_file": query["source_file"],
                "source_changed_statements": query["source_statements"],
                "candidate_target_files": candidates,
            }
        )
    shared_context = {
        "source_change": {
            "repository": source.repo,
            "title": (source.title or "")[:500],
            "body": (source.body or "")[:1000],
            "commit_message": (source.message or "")[:500],
        },
        "target_repository": state["target"].repo,
        "file_queries": prompt_queries,
    }
    action_payload = {
        "task": "Answer each named repository-graph action question for every source-file query.",
        **shared_context,
        "action_questions": {
            "AnalyzeSourceChange": "What behavior, symbols, and file role are changed by this source file?",
            "LocateFile": "Which candidate IDs match the source path, basename, extension, or directory role?",
            "LocateCounterpart": "Which existing candidate IDs implement the same responsibility as the source file?",
            "InferCounterpart": "If no existing counterpart is adequate, what source-derived target path should be considered?",
            "LocateSymbol": "Which candidate IDs contain target symbols corresponding to changed source symbols?",
            "LocateCodeSnippet": "Which candidate code previews contain source-like structures or operations?",
            "SemanticSearch": "Which candidates implement the source change intent despite lexical or path differences?",
            "ShowCode": "After inspecting the supplied target pre-fix previews, which candidates retain affected behavior?",
            "LocateRelatedArtifacts": "Which candidate IDs are related tests, fixtures, generated outputs, or data companions?",
        },
        "output_schema": {
            "action_answers": [
                {
                    "query_id": 0,
                    "source_intent": "concise answer to AnalyzeSourceChange",
                    "LocateFile": [0],
                    "LocateCounterpart": [0],
                    "InferCounterpart": "path or empty string",
                    "LocateSymbol": [0],
                    "LocateCodeSnippet": [0],
                    "SemanticSearch": [0],
                    "ShowCode": [0],
                    "LocateRelatedArtifacts": [],
                    "evidence": "brief grounded explanation",
                }
            ]
        },
        "constraints": [
            "Return one action answer for every query_id and every named action.",
            "For ID-valued answers, use only candidate IDs listed under that query.",
            "Use only source metadata and target pre-fix graph evidence supplied here.",
            "Return strict JSON only.",
        ],
    }
    action_system = (
        "You are a repository-graph QA agent for cross-repository patch propagation. "
        "Answer each graph action question using file-tree, AST-symbol, and code-preview evidence. "
        "Do not output chain-of-thought. Return JSON only."
    )
    action_response = llm.complete_json(action_system, action_payload)
    answers_by_query: dict[int, dict[str, Any]] = {}
    if action_response:
        for item in action_response.get("action_answers", []):
            try:
                query_id = int(item.get("query_id"))
            except (TypeError, ValueError):
                continue
            if not 0 <= query_id < len(queries):
                continue
            limit = len(queries[query_id]["candidates"])
            clean = dict(item)
            for action in (
                "LocateFile",
                "LocateCounterpart",
                "LocateSymbol",
                "LocateCodeSnippet",
                "SemanticSearch",
                "ShowCode",
                "LocateRelatedArtifacts",
            ):
                clean[action] = _valid_ids(item.get(action), limit)
            answers_by_query[query_id] = clean

    finalize_payload = {
        "task": "Finalize the affected target-file ranking from the completed graph-action answers.",
        **shared_context,
        "action_answers": [
            {"query_id": query_id, **answers_by_query.get(query_id, {})}
            for query_id in range(len(queries))
        ],
        "output_schema": {
            "rankings": [
                {
                    "query_id": 0,
                    "ranked_candidate_ids": [0, 1, 2],
                    "brief_rationale": "one short evidence-based sentence",
                }
            ]
        },
        "constraints": [
            "Return one ranking for every query_id.",
            "Use only candidate IDs listed under that query.",
            "Reconcile evidence across all action answers rather than relying on path alone.",
            "Return strict JSON only.",
        ],
    }
    finalize_system = (
        "You are the FinalizeResults agent in PropQA. Use the answers produced by each graph action "
        "to rank affected target files. Do not output chain-of-thought. Return JSON only."
    )
    final_response = llm.complete_json(finalize_system, finalize_payload)
    by_query: dict[str, list[int]] = {}
    rationales: dict[str, str] = {}
    if final_response:
        for item in final_response.get("rankings", []):
            try:
                query_id = int(item.get("query_id"))
            except (TypeError, ValueError):
                continue
            if not 0 <= query_id < len(queries):
                continue
            source_file = queries[query_id]["source_file"]
            by_query[source_file] = _valid_ids(
                item.get("ranked_candidate_ids"), len(queries[query_id]["candidates"])
            )
            rationales[source_file] = str(item.get("brief_rationale") or "")[:300]

    predictions = {}
    for query in queries:
        source_file = query["source_file"]
        ranked_ids = by_query.get(source_file, [])
        ranked_paths = [candidate_lookup[(source_file, candidate_id)] for candidate_id in ranked_ids]
        for item in query["candidates"]:
            if item["path"] not in ranked_paths:
                ranked_paths.append(item["path"])
        predictions[source_file] = ranked_paths[:5]
    return predictions, {
        "provider": llm.name,
        "model": llm.model,
        "base_url": llm.used_base_url,
        "llm_used": action_response is not None and final_response is not None,
        "llm_error": llm.last_error,
        "returned_queries": len(by_query),
        "expected_queries": len(queries),
        "rationales": rationales,
        "action_answers": answers_by_query,
        "action_qa_used": action_response is not None,
        "finalize_qa_used": final_response is not None,
    }


def baseline_rows() -> list[dict[str, Any]]:
    pure = json.loads(Path("outputs/rq5_filepair_repartitioned_v2_pure_nicad.json").read_text(encoding="utf-8"))
    filtered = json.loads(Path("outputs/path_prefiltered_baselines_rq5.json").read_text(encoding="utf-8"))
    rows = []
    for row in pure["file_pair_summary"]:
        if row["method"] in {"Open-NiCad", "code2vec"} and row["metric"] != "MeanRank":
            rows.append(dict(row))
    rename = {
        "Open-NiCad+path_prefilter": "Open-NiCad+Path",
        "code2vec+path_prefilter": "code2vec+Path",
    }
    for row in filtered["filepair_result"]["summary"]:
        if row["metric"] != "MeanRank":
            rows.append({**row, "method": rename[row["method"]]})
    return rows


def values_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        result.setdefault(row["method"], {})[row["metric"]] = float(row["value"])
    return result


def markdown(payload: dict[str, Any]) -> str:
    values = values_by_method(payload["rows"])
    lines = [
        "# LLM-in-the-loop File-level PropQA",
        "",
        f"Evaluation records: {payload['file_pair_records']}",
        "",
    ]
    lines += [
        "| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in BASELINES + ("PropQA",):
        row = values[method]
        lines.append(
            f"| {method} | {row['Hit@1']:.3f} | {row['Hit@3']:.3f} | "
            f"{row['Hit@5']:.3f} | {row['MRR@5']:.3f} |"
        )
    lines.append(f"LLM failures: {payload['llm_failures']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/URL_Results_detection_url_celo_v2.json")
    parser.add_argument("--output", default="outputs/file_llm_v3.json")
    parser.add_argument("--checkpoint", default=".cache/file_llm_v3.pkl")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--batch-queries", type=int, default=5)
    args = parser.parse_args()
    load_dotenv()
    pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    client = GitHubClient(Path(".cache/github"), token=os.environ.get("GITHUB_TOKEN"))
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    llm = LLMClient(
        "PropQA",
        os.environ.get("API_KEY", ""),
        os.environ.get("BASE_URL", ""),
        os.environ.get("MODEL", ""),
    )
    if not llm.api_key or not llm.base_url or not llm.model:
        raise RuntimeError("API_KEY, BASE_URL, and MODEL must be configured")
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.max_pairs:
        saved = pickle.loads(checkpoint.read_bytes())
        completed = saved["completed"]
        records = saved["records"]
        predictions = saved["predictions"]
        details = saved["details"]
        errors = saved["errors"]
        print(f"resuming after pair {completed}", flush=True)
    else:
        completed = 0
        records = []
        predictions = {}
        details = []
        errors = []

    for pair_index, pair in enumerate(pairs, start=1):
        if pair_index <= completed:
            continue
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, _oracle = split_detector_state(setup)
            pair_records = build_file_pair_records(pair_index, pair, setup)
            records.extend(pair_records)
            queries = []
            for source_file in sorted({record["source_file"] for record in pair_records}):
                source_state = restrict_state_to_source_file(state, source_file)
                candidates = file_action_rank(source_state, 20)
                for item in candidates[:10]:
                    item["target_symbols"] = candidate_symbols(runner, source_state, item["path"])
                    item["target_code_preview"] = candidate_code_preview(
                        runner, source_state, item["path"], item["target_symbols"]
                    )
                queries.append(
                    {
                        "source_file": source_file,
                        "source_statements": compact_source_statements(state, source_file),
                        "candidates": candidates,
                    }
                )
            pair_trace = {"index": pair_index, "llm": []}
            for start in range(0, len(queries), args.batch_queries):
                batch = queries[start : start + args.batch_queries]
                ranked, trace = rerank_batch(llm, state, batch)
                pair_trace["llm"].append(trace)
                for source_file, paths in ranked.items():
                    predictions[(pair_index, source_file, "PropQA")] = paths
            pair_trace["file_pairs"] = [
                {
                    "source_file": record["source_file"],
                    "target_file": record["target_file"],
                    "prediction": predictions.get(
                        (pair_index, record["source_file"], "PropQA"), []
                    ),
                }
                for record in pair_records
            ]
            details.append(pair_trace)
            print(f"[{pair_index}/{len(pairs)}] ok queries={len(queries)}", flush=True)
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
                        "predictions": predictions,
                        "details": details,
                        "errors": errors,
                    }
                )
            )

    graphqa_rows = file_metrics_for_records(
        records,
        predictions,
        ["PropQA"],
    )
    rows = baseline_rows() + graphqa_rows
    llm_failures = sum(
        not trace["llm_used"] or trace["returned_queries"] != trace["expected_queries"]
        for detail in details
        for trace in detail["llm"]
    )
    payload = {
        "dataset": args.dataset,
        "evaluated_pairs": len(pairs) - len(errors),
        "file_pair_records": len(records),
        "llm": {"model": llm.model, "used_base_url": llm.used_base_url},
        "rows": rows,
        "llm_failures": llm_failures,
        "details": details,
        "errors": errors,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
