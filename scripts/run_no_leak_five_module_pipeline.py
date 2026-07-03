#!/usr/bin/env python3
"""No-leak five-module file/statement detection pipeline.

This script is the new canonical experimental entrypoint. Target PR/commit
metadata and target changed files/statements are used only for evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_detection_tables import format_float
from no_leak_ground_truth import StatementGT, line_hits_statement_gt, statement_ground_truth_from_files
from propagation_detector import GitHubClient, changed_old_line_ranges
from run_agent_strategy_matrix import map_candidates
from run_contextual_qa_judge import build_prediction, candidate_files, candidate_statements, clip, llm_judge
from run_file_agent_llm_ablation import collect_file_action_candidates
from run_llm_in_loop_hard_cases import LoopLLM, prepare_state
from run_new_hard_baseline_ablation import (
    Embedder,
    action_deletion_interface_build_impact,
    action_graphqa,
    action_symbol_impact,
    action_test_impact,
    baseline_open_nicad,
    baseline_semantic_embedding,
    path_prefilter,
)
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv
from run_qa_mcts_small_sample import prefer_side_files


MODULES = ["graphqa", "nicad", "semantic_embedding", "dependency", "llm_fusion", "weighted_fusion"]
OLD_HARD_ORIGINAL_INDICES = {4, 16, 83, 93, 126, 127, 129, 130, 132, 133, 135, 171, 196}
ORACLE_KEYS = {"target_changed_paths", "target_files", "target_ranges"}


def noleak_source_context(pair: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    source = pair.get("Source") or {}
    auxiliary = [path for path in source.get("auxiliary_file_names", []) if path.rsplit("/", 1)[-1].lower() != ".gitignore"]
    return {
        "repo": state["propagator"].repo,
        "title": clip(source.get("title") or source.get("message") or "", 500),
        "body": clip(source.get("body") or "", 1200),
        "message": clip(source.get("message") or "", 800),
        "changed_files": state.get("source_changed_paths", [])[:30],
        "primary_changed_files": state.get("source_changed_paths", [])[:30],
        "auxiliary_changed_files": auxiliary[:30],
        "file_role_note": "Primary files drive code localization; auxiliary files are handled by a separate propagation action.",
    }


def split_detector_state(setup: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Physically separate prediction inputs from target evaluation oracle."""
    oracle = {key: setup.get(key) for key in ORACLE_KEYS}
    detector_state = {key: value for key, value in setup.items() if key not in ORACLE_KEYS}
    target = detector_state["target"]
    detector_state["target"] = dataclasses.replace(
        target,
        number=None,
        title="",
        body="",
        message="",
        url="",
        merged_at="",
        merged_sha="",
        file_names=[],
    )
    assert_no_file_oracle(detector_state)
    return detector_state, oracle


def normalize_target_oracle_to_dataset_files(setup: dict[str, Any]) -> dict[str, Any]:
    """Restrict evaluation oracle to the dataset's curated target file list."""
    target_files = prefer_side_files(setup["target_files"], setup["target"].file_names)
    setup = dict(setup)
    setup["target_files"] = target_files
    setup["target_changed_paths"] = [item.get("filename") for item in target_files if item.get("filename")]
    setup["target_ranges"] = changed_old_line_ranges(target_files)
    return setup


def assert_no_file_oracle(state: dict[str, Any]) -> None:
    leaked = ORACLE_KEYS & set(state)
    if leaked:
        raise RuntimeError(f"file detector state contains forbidden oracle keys: {sorted(leaked)}")
    target = state.get("target")
    if target and any([target.title, target.body, target.message, target.url, target.merged_sha, target.file_names]):
        raise RuntimeError("file detector state contains forbidden target change metadata")


def normalize_module_files(files: list[dict[str, Any]], module: str, limit: int) -> list[dict[str, Any]]:
    out = []
    for rank, item in enumerate(files[:limit], start=1):
        out.append(
            {
                "path": item["path"],
                "score": float(item.get("score", item.get("confidence", 0.0)) or 0.0),
                "rank": rank,
                "module": module,
                "matched_source_path": item.get("matched_source_path"),
                "evidence": item.get("evidence", {}),
                "motivation": item.get("motivation") or short_motivation(module, item),
            }
        )
    return out


def short_motivation(module: str, item: dict[str, Any]) -> str:
    path = item.get("path", "")
    if module == "graphqa":
        return f"Graph/path/AST evidence links source changes to `{path}`."
    if module == "nicad":
        return f"Normalized clone evidence suggests source patch code is similar to `{path}`."
    if module == "semantic_embedding":
        return f"Independent file-content embedding retrieval ranks `{path}` as semantically close."
    if module == "dependency":
        return f"Call/reference, test, deletion, interface, or build-impact evidence points to `{path}`."
    return f"Candidate `{path}` from {module}."


def graphqa_llm_expand(
    llm: LoopLLM,
    pair: dict[str, Any],
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Use question-guided LLM exploration to promote graph candidates.

    The LLM can only select paths from the pre-fix target graph candidate pool.
    Target change metadata is never included.
    """
    if not llm.enabled:
        return candidates[: args.file_top]
    by_path = {item["path"]: item for item in candidates}
    payload = {
        "task": "PropQA exploration: rank target-repository files that may be affected by the source change.",
        "source": noleak_source_context(pair, state),
        "target": {"repo": state["target"].repo, "candidate_paths": [item["path"] for item in candidates[: args.graphqa_llm_pool]]},
        "questions": [
            "For each source changed file, which target candidate is its structural, renamed, or moved counterpart?",
            "Which test, config, build, generated, or dependency companion files may also require propagation?",
            "Which candidates may be affected through function, interface, variable, or API references?",
            "Which candidates are easy to miss if ranked only by global path similarity?",
            "Use primary_changed_files for the primary ranking; do not mix auxiliary_changed_files into the primary list.",
        ],
        "constraints": [
            "Use only candidate_paths.",
            "Do not assume any target PR, commit, message, or known target changed file.",
            "Return concise strict JSON.",
        ],
        "output_schema": {"target_files": [{"path": "candidate path", "reason": "short graph/impact reason"}]},
    }
    result = llm.complete_json_with_system("You are a no-leak PropQA repository exploration agent.", payload)
    if not result:
        return candidates[: args.file_top]
    # Preserve the strong deterministic PropQA head. LLM exploration expands
    # the candidate frontier instead of freely replacing the reliable Top-10.
    preserve = min(args.graphqa_preserve_top, args.file_top)
    selected = [dict(item) for item in candidates[:preserve]]
    seen = {item["path"] for item in selected}
    for item in result.get("target_files", []):
        path = item.get("path") if isinstance(item, dict) else None
        if path not in by_path or path in seen:
            continue
        candidate = dict(by_path[path])
        candidate["graphqa_llm_reason"] = item.get("reason", "")
        candidate["graphqa_expansion"] = True
        selected.append(candidate)
        seen.add(path)
        if len(selected) >= args.file_top:
            break
    for item in candidates:
        if len(selected) >= args.file_top:
            break
        if item["path"] not in seen:
            selected.append(item)
            seen.add(item["path"])
    return selected


def auxiliary_file_type(path: str) -> str:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if name == "go.sum":
        return "dependency_lock"
    if name.endswith((".md", ".rst", ".adoc")) or "/doc/" in lower or "/docs/" in lower:
        return "documentation"
    if "/testdata/" in lower or "/fixture" in lower or name.endswith((".snap", ".golden", ".out", ".rlp", ".hex")):
        return "snapshot_or_test_output"
    return "comment_or_other_auxiliary"


def action_find_auxiliary_propagation(pair: dict[str, Any], state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Retrieve auxiliary counterparts from the target pre-fix tree."""
    source = pair.get("Source") or {}
    source_aux = [
        path
        for path in source.get("auxiliary_file_names", [])
        if path.rsplit("/", 1)[-1].lower() != ".gitignore"
    ]
    candidates: dict[str, dict[str, Any]] = {}
    target_paths = set(state["target_tree"])
    for source_path in source_aux:
        source_name = source_path.rsplit("/", 1)[-1].lower()
        source_dir = source_path.rsplit("/", 1)[0].lower() if "/" in source_path else ""
        for target_path in state["target_tree"]:
            if target_path.rsplit("/", 1)[-1].lower() == ".gitignore":
                continue
            target_name = target_path.rsplit("/", 1)[-1].lower()
            target_dir = target_path.rsplit("/", 1)[0].lower() if "/" in target_path else ""
            exact = source_path == target_path
            same_name = source_name == target_name
            same_dir = bool(source_dir and source_dir == target_dir)
            if not exact and not same_name:
                continue
            score = 1.0 if exact else 0.78 + (0.08 if same_dir else 0.0)
            old = candidates.get(target_path)
            item = {
                "path": target_path,
                "score": score,
                "matched_source_path": source_path,
                "auxiliary_type": auxiliary_file_type(source_path),
                "evidence": {"exact_path": exact, "same_filename": same_name, "same_directory": same_dir},
                "action": "FindAuxiliaryPropagation",
            }
            if old is None or score > old["score"]:
                candidates[target_path] = item
        if source_path not in target_paths:
            # A propagated snapshot, fixture, documentation, or lock file may
            # need to be created in the target. This path is derived only from
            # the Source auxiliary change, not from target change metadata.
            candidates[source_path] = {
                "path": source_path,
                "score": 0.72,
                "matched_source_path": source_path,
                "auxiliary_type": auxiliary_file_type(source_path),
                "evidence": {
                    "exact_path": False,
                    "same_filename": True,
                    "same_directory": True,
                    "proposed_new_file": True,
                    "path_source": "source_auxiliary_change",
                },
                "action": "ProposeNewAuxiliaryFile",
            }
    return sorted(candidates.values(), key=lambda item: (-item["score"], item["path"]))[:limit]


def llm_validate_auxiliary_propagation(
    llm: LoopLLM,
    pair: dict[str, Any],
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if not candidates or not llm.enabled:
        return candidates[:limit]
    payload = {
        "task": "Validate auxiliary-file propagation candidates separately from primary code localization.",
        "source": noleak_source_context(pair, state),
        "target": {"repo": state["target"].repo},
        "candidates": [
            {
                "path": item["path"],
                "matched_source_path": item["matched_source_path"],
                "auxiliary_type": item["auxiliary_type"],
                "evidence": item["evidence"],
            }
            for item in candidates
        ],
        "questions": [
            "Is this an exact or moved auxiliary counterpart that should accompany the propagated source change?",
            "If the path does not exist in the pre-fix target tree, should the same auxiliary path be created?",
            "Is it a dependency lock, documentation/comment companion, or snapshot/test-output companion?",
            "Should it be retained for auxiliary propagation review while remaining outside primary code metrics?",
        ],
        "constraints": [
            "Select only provided candidates.",
            "Do not use or infer target PR/commit metadata or known changed files.",
            "Exclude .gitignore.",
            "Return strict JSON only.",
        ],
        "output_schema": {"auxiliary_files": [{"path": "candidate path", "confidence": 0.0, "reason": "short"}]},
    }
    result = llm.complete_json_with_system("You are a no-leak auxiliary propagation QA agent.", payload)
    if not result:
        return candidates[:limit]
    by_path = {item["path"]: item for item in candidates}
    selected, seen = [], set()
    for item in result.get("auxiliary_files", []):
        path = item.get("path") if isinstance(item, dict) else None
        if path not in by_path or path in seen:
            continue
        out = dict(by_path[path])
        out["llm_confidence"] = float(item.get("confidence", 0.0) or 0.0)
        out["llm_reason"] = item.get("reason", "")
        selected.append(out)
        seen.add(path)
    for item in candidates:
        if len(selected) >= limit:
            break
        if item["path"] not in seen:
            selected.append(item)
            seen.add(item["path"])
    return selected[:limit]


def file_module_outputs(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    llm: LoopLLM,
    embedder: Embedder,
    pair: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    # Module 1: PropQA over file tree/path/AST structure only.
    graph_pool = path_prefilter(state, args.graphqa_pool)
    graph = graphqa_llm_expand(llm, pair, state, graph_pool, args)
    # Module 2: normalized clone retrieval, used mainly for easy direct ports.
    nicad, _ = baseline_open_nicad(state, runner, args)
    # Module 3: independent semantic retrieval over every target code file.
    semantic_embedding, _ = baseline_semantic_embedding(state, runner, args, embedder)
    # Module 4: dependency/propagation impact. This starts from an empty seed so
    # it does not collapse into the PropQA path-ranker.
    dependency = action_symbol_impact(state, [], args.file_top * 2)
    dependency = action_deletion_interface_build_impact(state, dependency, args.file_top * 2)
    dependency = action_test_impact(state, dependency, args.file_top * 2)
    return {
        "graphqa": normalize_module_files(graph, "graphqa", args.file_top),
        "nicad": normalize_module_files(nicad, "nicad", args.file_top),
        "semantic_embedding": normalize_module_files(semantic_embedding, "semantic_embedding", args.file_top),
        "dependency": normalize_module_files(dependency, "dependency", args.file_top),
    }


def weighted_file_fusion(module_outputs: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    weights = {"graphqa": 0.30, "nicad": 0.20, "semantic_embedding": 0.30, "dependency": 0.20}
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, dict[str, Any]] = defaultdict(dict)
    matched: dict[str, str | None] = {}
    for module, files in module_outputs.items():
        if not files:
            continue
        max_score = max(float(f["score"]) for f in files) or 1.0
        for item in files:
            norm = float(item["score"]) / max_score
            rank_bonus = 1.0 / max(int(item["rank"]), 1)
            scores[item["path"]] += weights.get(module, 0.1) * (0.75 * norm + 0.25 * rank_bonus)
            evidence[item["path"]][module] = {"rank": item["rank"], "score": item["score"], "motivation": item["motivation"]}
            matched.setdefault(item["path"], item.get("matched_source_path"))
    return [
        {"path": path, "score": round(score, 6), "evidence": evidence[path], "matched_source_path": matched.get(path)}
        for path, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: args.file_top]
    ]


def llm_file_fusion(llm: LoopLLM, pair: dict[str, Any], state: dict[str, Any], module_outputs: dict[str, list[dict[str, Any]]], weighted: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not llm.enabled:
        return weighted
    all_candidates = {item["path"]: item for files in module_outputs.values() for item in files}
    payload = {
        "task": "Fuse no-leak file candidates for cross-repository propagation detection.",
        "source": noleak_source_context(pair, state),
        "target_available": {
            "repo": state["target"].repo,
            "note": "Target PR/commit title, body, message and changed files are hidden. Use only candidate evidence.",
        },
        "module_outputs": {
            name: [
                {
                    "path": item["path"],
                    "rank": item["rank"],
                    "score": item["score"],
                    "motivation": item["motivation"],
                    "matched_source_path": item.get("matched_source_path"),
                }
                for item in files[: args.llm_module_items]
            ]
            for name, files in module_outputs.items()
        },
        "weighted_fusion_seed": [{"path": item["path"], "score": item["score"]} for item in weighted[: args.llm_module_items]],
        "output_schema": {"target_files": [{"path": "candidate path", "confidence": 0.0, "reason": "short"}]},
        "constraints": [
            "Return strict JSON only.",
            "Select only paths appearing in module_outputs or weighted_fusion_seed.",
            "Do not use or infer target PR/commit metadata.",
            "Prefer files supported by multiple independent modules.",
        ],
    }
    system = "You are a no-leak file fusion judge for repository propagation detection."
    result = llm.complete_json_with_system(system, payload)
    if not result:
        return weighted
    # The deterministic PropQA head is currently the strongest Top-10 file
    # ranker. Keep it stable and use LLM/module evidence to expand positions
    # 11-20. Promotion into Top-10 should be evaluated as a separate, stricter
    # validation action rather than an unconstrained rerank.
    graph_head = module_outputs.get("graphqa", [])[: min(args.final_preserve_top, args.file_top)]
    selected = [dict(item) for item in graph_head]
    seen = {item["path"] for item in selected}
    for item in result.get("target_files", []):
        path = item.get("path") if isinstance(item, dict) else None
        if path in all_candidates and path not in seen:
            base = dict(all_candidates[path])
            base["llm_confidence"] = float(item.get("confidence", 0.0) or 0.0)
            base["llm_reason"] = item.get("reason", "")
            base["score"] = base.get("score", 0.0) + 2.0 * base["llm_confidence"]
            selected.append(base)
            seen.add(path)
    for item in weighted:
        if len(selected) >= args.file_top:
            break
        if item["path"] not in seen:
            selected.append(item)
            seen.add(item["path"])
    return selected[: args.file_top]


def statement_module_outputs(state: dict[str, Any], runner: PropagationQAMCTS, file_rank: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    graph_stmts = map_candidates(runner, state, file_rank[: args.inspect_files], args.statement_candidates)
    # Text/clone and code2vec proxy currently reuse the same mapper but rerank by
    # different available evidence fields.
    text_stmts = sorted(graph_stmts, key=lambda x: (-float(x.get("line_similarity", x.get("score", 0.0))), x["path"], int(x["line"])))
    c2v_stmts = sorted(graph_stmts, key=lambda x: (-(float(x.get("symbol_score", 0.0)) + float(x.get("score", 0.0))), x["path"], int(x["line"])))
    impact_stmts = sorted(
        graph_stmts,
        key=lambda x: (
            -float(x.get("pattern_score", 0.0)),
            -float(x.get("symbol_score", 0.0)),
            x["path"],
            int(x["line"]),
        ),
    )
    return {
        "stmt_graphqa": graph_stmts[: args.statement_top],
        "stmt_text_clone": text_stmts[: args.statement_top],
        "stmt_code2vec": c2v_stmts[: args.statement_top],
        "stmt_dependency": impact_stmts[: args.statement_top],
    }


def weighted_statement_fusion(module_outputs: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    weights = {"stmt_graphqa": 0.28, "stmt_text_clone": 0.22, "stmt_code2vec": 0.28, "stmt_dependency": 0.22}
    scores: dict[tuple[str, int], float] = defaultdict(float)
    items: dict[tuple[str, int], dict[str, Any]] = {}
    for module, stmts in module_outputs.items():
        for rank, item in enumerate(stmts, start=1):
            try:
                key = (item["path"], int(item["line"]))
            except Exception:
                continue
            scores[key] += weights.get(module, 0.1) * (float(item.get("score", 0.0)) + 1.0 / rank)
            old = items.setdefault(key, dict(item))
            old.setdefault("statement_modules", []).append(module)
    ranked = []
    for key, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[: args.statement_top]:
        item = dict(items[key])
        item["score"] = round(score, 6)
        ranked.append(item)
    return ranked


def llm_statement_fusion(llm: LoopLLM, pair: dict[str, Any], state: dict[str, Any], file_rank: list[dict[str, Any]], stmt_outputs: dict[str, list[dict[str, Any]]], weighted: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not llm.enabled:
        return weighted
    agent_results = []
    agent_results.append({"method": "file_fusion", "prediction": {"target_files": file_rank[: args.file_top], "target_statements": weighted[: args.statement_top]}})
    for name, stmts in stmt_outputs.items():
        agent_results.append({"method": name, "prediction": {"target_files": file_rank[: args.file_top], "target_statements": stmts[: args.statement_top]}})
    files = candidate_files(*agent_results)
    statements = candidate_statements(*agent_results)
    llm_json, _ = llm_judge(llm, pair, state, files, statements, hide_target_metadata=True)
    prediction = build_prediction(llm_json, files, statements, args.file_top, args.statement_top)
    return prediction.get("target_statements", weighted)[: args.statement_top]


def file_metrics(predicted: list[str], gt: set[str], k: int) -> dict[str, float]:
    preds = predicted[:k]
    hits = set(p for p in preds if p in gt)
    precision = len(hits) / len(preds) if preds else 0.0
    recall = len(hits) / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rr = 0.0
    for rank, path in enumerate(preds, start=1):
        if path in gt:
            rr = 1.0 / rank
            break
    return {"precision": precision, "recall": recall, "f1": f1, "mrr": rr}


def stmt_metrics(predicted: list[dict[str, Any]], gt_items: list[StatementGT], k: int, tolerance: int) -> dict[str, float]:
    preds = predicted[:k]
    hit_pred = 0
    covered = set()
    rr = 0.0
    for rank, item in enumerate(preds, start=1):
        try:
            path, line = item["path"], int(item["line"])
        except Exception:
            continue
        matched = [idx for idx, gt in enumerate(gt_items) if line_hits_statement_gt(path, line, gt, tolerance)]
        if matched:
            hit_pred += 1
            covered.update(matched)
            if rr == 0.0:
                rr = 1.0 / rank
    precision = hit_pred / len(preds) if preds else 0.0
    recall = len(covered) / len(gt_items) if gt_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "coverage": recall, "f1": f1, "mrr": rr}


def aggregate(rows: list[dict[str, Any]], level: str, k: int) -> dict[str, Any]:
    subset = [r for r in rows if r["level"] == level and r["k"] == k]
    n = len(subset)
    out = {"level": level, "k": k, "pairs": n}
    for key in ["precision", "recall", "coverage", "f1", "mrr"]:
        out[key] = sum(r.get(key, r.get("recall", 0.0)) for r in subset) / n if n else 0.0
    return out


def oracle_file_rank_for_statement_localization(state: dict[str, Any], target_changed_paths: list[str]) -> list[dict[str, Any]]:
    """Statement localization is evaluated independently of file detection.

    The target changed files are oracle scope for this stage only; target
    statements remain hidden from prediction and are used after prediction for
    evaluation.
    """
    source_hint = next(iter(state.get("source_changed_paths", [])), None)
    return [
        {"path": path, "score": 1.0, "matched_source_path": source_hint, "module": "oracle_statement_scope"}
        for path in target_changed_paths
    ]


def run_pair(index: int, pair: dict[str, Any], client: GitHubClient, llm: LoopLLM, embedder: Embedder, args: argparse.Namespace) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=args.file_top, top_statements=args.statement_top)
    setup = prepare_state(runner, pair)
    if setup.get("status") and setup.get("status") != "ok":
        return {"index": index, **setup}
    setup = normalize_target_oracle_to_dataset_files(setup)
    state, oracle = split_detector_state(setup)
    source_summary = runner._execute(state, {"type": "InspectSourcePatch"})
    runner._update_state(state, {"type": "InspectSourcePatch"}, source_summary)

    assert_no_file_oracle(state)
    modules = file_module_outputs(state, runner, llm, embedder, pair, args)
    weighted_files = weighted_file_fusion(modules, args)
    llm_files = llm_file_fusion(llm, pair, state, modules, weighted_files, args)
    auxiliary_candidates = action_find_auxiliary_propagation(pair, state, args.auxiliary_top)
    auxiliary_files = llm_validate_auxiliary_propagation(llm, pair, state, auxiliary_candidates, args.auxiliary_top)
    statement_scope_files = oracle_file_rank_for_statement_localization(state, oracle["target_changed_paths"])
    stmt_modules = statement_module_outputs(state, runner, statement_scope_files, args)
    weighted_stmts = weighted_statement_fusion(stmt_modules, args)
    llm_stmts = llm_statement_fusion(llm, pair, state, statement_scope_files, stmt_modules, weighted_stmts, args)

    target_gt_files = set(oracle["target_changed_paths"])
    target_gt_statements = statement_ground_truth_from_files(oracle["target_files"])
    metrics = []
    for k in args.file_eval_ks:
        metrics.append({"level": "file", "k": k, **file_metrics([x["path"] for x in llm_files], target_gt_files, k)})
    target_auxiliary_files = {
        path
        for path in (pair.get("Infestor") or {}).get("auxiliary_file_names", [])
        if path.rsplit("/", 1)[-1].lower() != ".gitignore"
    }
    for k in args.auxiliary_eval_ks:
        metrics.append({"level": "auxiliary_file", "k": k, **file_metrics([x["path"] for x in auxiliary_files], target_auxiliary_files, k)})
    for k in args.statement_eval_ks:
        metrics.append({"level": "statement", "k": k, **stmt_metrics(llm_stmts, target_gt_statements, k, args.statement_tolerance)})
    return {
        "index": index,
        "status": "ok",
        "prediction": {
            "target_files": llm_files,
            "target_auxiliary_files": auxiliary_files,
            "target_statements": llm_stmts,
            "strategy": "no_leak_graphqa_clone_semantic_dependency_pipeline",
            "statement_scope": "oracle_target_changed_files_for_independent_localization",
        },
        "file_module_outputs": modules,
        "auxiliary_action_output": auxiliary_candidates,
        "statement_module_outputs": stmt_modules,
        "ground_truth": {
            "target_changed_files": sorted(target_gt_files),
            "target_auxiliary_files": sorted(target_auxiliary_files),
            "statement_ground_truth": [item.__dict__ for item in target_gt_statements],
        },
        "metrics": metrics,
    }


def split_label(pair: dict[str, Any]) -> str:
    difficulty = ((pair.get("_manual_curation") or {}).get("difficulty") or "").lower()
    origin = pair.get("_combined_hard_origin")
    original_index = pair.get("original_index")
    dataset_index = pair.get("_dataset_index")
    if difficulty == "hard" or origin or original_index in OLD_HARD_ORIGINAL_INDICES or dataset_index in OLD_HARD_ORIGINAL_INDICES:
        return "hard"
    return "simple"


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"), sleep_seconds=args.sleep)
    llm = LoopLLM(timeout=args.llm_timeout)
    embedder = Embedder(args.embedding_model)
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    results = []
    done_indices = set()
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            results = list(existing.get("results") or [])
            done_indices = {int(r["index"]) for r in results if "index" in r}
            print(f"[resume] loaded {len(results)} existing results from {out_path}", flush=True)
        except Exception as exc:
            print(f"[resume] ignored unreadable existing output: {type(exc).__name__}: {exc}", flush=True)
    for index, pair in enumerate(pairs, start=1):
        if index in done_indices:
            continue
        try:
            result = run_pair(index, pair, client, llm, embedder, args)
        except Exception as exc:
            result = {"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        result["dataset_split"] = split_label(pair)
        results.append(result)
        out_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}/{len(pairs)}] {result['status']}", flush=True)

    results.sort(key=lambda r: int(r.get("index", 0)))
    rows = []
    for split in ["all", "simple", "hard"]:
        split_results = [r for r in results if r.get("status") == "ok" and (split == "all" or r.get("dataset_split") == split)]
        metric_rows = [m for r in split_results for m in r["metrics"]]
        for level in ["file", "auxiliary_file", "statement"]:
            eval_ks = args.file_eval_ks if level == "file" else args.auxiliary_eval_ks if level == "auxiliary_file" else args.statement_eval_ks
            for k in eval_ks:
                row = aggregate(metric_rows, level, k)
                row["dataset"] = split
                row["method"] = args.method_name
                rows.append(row)
    return {
        "summary": {
            "input": args.input,
            "pairs": len(pairs),
            "ok": sum(1 for r in results if r.get("status") == "ok"),
            "llm_enabled": llm.enabled,
            "hide_target_metadata": True,
            "provider": args.provider,
        },
        "rows": rows,
        "results": results,
    }


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Method | Level | K | Pairs | Precision | Recall/Coverage | F1 | MRR |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        recall = r.get("coverage", r.get("recall", 0.0)) if r["level"] == "statement" else r.get("recall", 0.0)
        lines.append(
            f"| {r['dataset']} | {r['method']} | {r['level']} | {r['k']} | {r['pairs']} | "
            f"{r['precision']:.3f} | {recall:.3f} | {r['f1']:.3f} | {r['mrr']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/URL_Results_detection_primary_v4.json")
    parser.add_argument("--output", default="outputs/no_leak_five_module_pipeline.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--embedding-model", default="D:/Python/Blockchain/sentenceTransformer/all-MiniLM-L12-v2")
    parser.add_argument("--semantic-embedding-cache", default=".cache/semantic_embeddings")
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument("--semantic-max-target-files", type=int, default=0, help="0 means all target code files.")
    parser.add_argument("--local-git-repo-cache", default=".cache/git_repos")
    parser.add_argument("--local-git-archive-timeout", type=int, default=1800)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--method-name", default="NoLeak-FiveModule-QA")
    parser.add_argument("--file-top", type=int, default=20)
    parser.add_argument("--statement-top", type=int, default=100)
    parser.add_argument("--candidate-files", type=int, default=28)
    parser.add_argument("--expanded-files", type=int, default=56)
    parser.add_argument("--prefilter-files", type=int, default=80)
    parser.add_argument("--inspect-files", type=int, default=18)
    parser.add_argument("--candidate-statements", type=int, default=200)
    parser.add_argument("--llm-module-items", type=int, default=20)
    parser.add_argument("--graphqa-pool", type=int, default=120)
    parser.add_argument("--graphqa-llm-pool", type=int, default=80)
    parser.add_argument("--graphqa-preserve-top", type=int, default=10)
    parser.add_argument("--final-preserve-top", type=int, default=10)
    parser.add_argument("--statement-candidates", type=int, default=200)
    parser.add_argument("--statement-tolerance", type=int, default=2)
    parser.add_argument("--file-eval-ks", type=int, nargs="*", default=[3, 5, 8, 10, 20])
    parser.add_argument("--auxiliary-top", type=int, default=20)
    parser.add_argument("--auxiliary-eval-ks", type=int, nargs="*", default=[10, 20])
    parser.add_argument("--statement-eval-ks", type=int, nargs="*", default=[20, 50, 100])
    parser.add_argument("--max-file-chars", type=int, default=12000)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Compatibility with reusable baseline helpers.
    args.top_files = args.file_top
    args.top_statements = args.statement_top
    args.eval_ks = sorted(set(args.file_eval_ks + args.statement_eval_ks))
    payload = run(args)
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(
        "# No-Leak Five-Module Pipeline Results\n\n"
        + f"- Provider: {payload['summary']['provider']}\n"
        + f"- Pairs: {payload['summary']['pairs']}\n"
        + f"- OK: {payload['summary']['ok']}\n"
        + "- Target PR/commit metadata hidden: yes\n\n"
        + table(payload["rows"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
