#!/usr/bin/env python3
"""File-to-file propagation detection experiment.

This scenario treats each source changed file as an independent query and asks
the detector to rank target pre-fix files.  The file-pair ground truth is
constructed from curated source/target changed files by matching target files
to the most likely source file using path and patch evidence, then adding any
uncovered source files with their best target file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from propagation_detector import GitHubClient, jaccard, path_score, path_tokens, token_similarity
from run_new_hard_baseline_ablation import (
    action_graphqa,
    changed_symbol_terms,
    normalize_code,
    path_prefilter,
    safe_file_at,
    source_patch_text,
    source_terms,
    tokens,
)
from run_no_leak_five_module_pipeline import normalize_target_oracle_to_dataset_files
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv


def file_patch(files: list[dict[str, Any]], path: str) -> str:
    for item in files:
        if item.get("filename") == path:
            return item.get("patch") or ""
    return ""


def mapping_score(source_path: str, target_path: str, source_patch: str, target_patch: str) -> dict[str, float]:
    basename = 1.0 if source_path.rsplit("/", 1)[-1].lower() == target_path.rsplit("/", 1)[-1].lower() else 0.0
    path_sim = path_score(source_path, target_path)
    token_sim = jaccard(path_tokens(source_path), path_tokens(target_path))
    patch_sim = token_similarity(normalize_code(source_patch), normalize_code(target_patch))
    score = 0.42 * path_sim + 0.36 * patch_sim + 0.16 * basename + 0.06 * token_sim
    return {
        "score": round(score, 6),
        "path_score": round(path_sim, 6),
        "patch_score": round(patch_sim, 6),
        "basename": basename,
        "token_jaccard": round(token_sim, 6),
    }


def build_file_pair_records(index: int, pair: dict[str, Any], setup: dict[str, Any]) -> list[dict[str, Any]]:
    source_paths = list(dict.fromkeys(setup.get("source_changed_paths") or []))
    target_paths = list(dict.fromkeys(setup.get("target_changed_paths") or []))
    if not source_paths or not target_paths:
        return []
    scores: dict[tuple[str, str], dict[str, float]] = {}
    for sp in source_paths:
        spatch = file_patch(setup["source_files"], sp)
        for tp in target_paths:
            tpatch = file_patch(setup["target_files"], tp)
            scores[(sp, tp)] = mapping_score(sp, tp, spatch, tpatch)

    selected: set[tuple[str, str]] = set()
    # Ensure every target file has a source file.
    for tp in target_paths:
        best_sp = max(source_paths, key=lambda sp: (scores[(sp, tp)]["score"], sp))
        selected.add((best_sp, tp))
    # Ensure every source file has a target file.
    for sp in source_paths:
        if not any(s == sp for s, _ in selected):
            best_tp = max(target_paths, key=lambda tp: (scores[(sp, tp)]["score"], tp))
            selected.add((sp, best_tp))

    source = pair.get("Source") or {}
    target = pair.get("Infestor") or {}
    records = []
    for local_id, (sp, tp) in enumerate(sorted(selected), start=1):
        evidence = scores[(sp, tp)]
        records.append(
            {
                "file_pair_id": f"{index}:{local_id}:{sp}=>{tp}",
                "pair_index": index,
                "original_index": pair.get("original_index"),
                "source_repo": source.get("project"),
                "target_repo": target.get("project"),
                "source_type": source.get("type"),
                "target_type": target.get("type"),
                "source_number": source.get("pr_number") or source.get("number") or source.get("sha"),
                "target_number": target.get("pr_number") or target.get("number") or target.get("sha"),
                "source_file": sp,
                "target_file": tp,
                "mapping_score": evidence["score"],
                "mapping_evidence": evidence,
            }
        )
    return records


def restrict_state_to_source_file(state: dict[str, Any], source_file: str) -> dict[str, Any]:
    new_state = dict(state)
    new_state["source_changed_paths"] = [source_file]
    new_state["source_files"] = [item for item in state.get("source_files", []) if item.get("filename") == source_file]
    new_state["source_statements"] = [stmt for stmt in state.get("source_statements", []) if getattr(stmt, "path", None) == source_file]
    return new_state


def target_file_text(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace, path: str, cache: dict[tuple[str, str, str], str]) -> str:
    repo = state["target"].repo
    sha = state["target"].base_sha
    key = (repo, sha, path)
    if key in cache:
        return cache[key]
    git_dir = Path(getattr(args, "local_git_repo_cache", ".cache/git_repos")) / f"{repo.replace('/', '_')}.git"
    text = ""
    if git_dir.exists():
        try:
            result = subprocess.run(
                ["git", f"--git-dir={git_dir}", "show", f"{sha}:{path}"],
                check=True,
                capture_output=True,
                timeout=args.git_show_timeout,
            )
            text = result.stdout.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    if not text and args.allow_github_file_fetch:
        text = safe_file_at(runner, repo, sha, path)
    cache[key] = text or ""
    return cache[key]


def source_file_query_text(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace, source_file: str) -> str:
    repo = state["propagator"].repo
    sha = state["propagator"].merged_sha
    text = ""
    git_dir = Path(getattr(args, "local_git_repo_cache", ".cache/git_repos")) / f"{repo.replace('/', '_')}.git"
    if git_dir.exists():
        try:
            result = subprocess.run(
                ["git", f"--git-dir={git_dir}", "show", f"{sha}:{source_file}"],
                check=True,
                capture_output=True,
                timeout=args.git_show_timeout,
            )
            text = result.stdout.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    if not text and args.allow_github_file_fetch:
        text = safe_file_at(runner, repo, sha, source_file)
    if not text:
        text = "\n".join(stmt.text for stmt in state.get("source_statements", []) if getattr(stmt, "path", None) == source_file)
    return source_file + "\n" + (text or source_patch_text(state))[: args.max_file_chars]


def graphqa_files(state: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return action_graphqa(state, args.candidate_files)


def graphqa_action_files(state: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Add a no-leak new-file counterpart proposal to PropQA."""
    ranked = graphqa_files(state, args)
    source_file = state["source_changed_paths"][0]
    if source_file not in {item["path"] for item in ranked}:
        ranked = [
            {
                "path": source_file,
                "score": 1.2,
                "matched_source_path": source_file,
                "action": "ProposeNewCounterpart",
                "evidence": {
                    "path_source": "source_changed_file",
                    "target_pre_fix_node_required": False,
                },
            }
        ] + ranked
    else:
        for item in ranked:
            if item["path"] == source_file:
                item.setdefault("action", "ExactCounterpart")
    return ranked[: args.candidate_files]


def graphqa_impact_files(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    args: argparse.Namespace,
    content_cache: dict[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    """Expand PropQA with target files that reference changed source intent."""
    base = graphqa_action_files(state, args)
    base_rank = {item["path"]: rank for rank, item in enumerate(base, 1)}
    source_symbols = changed_symbol_terms(state)
    patch_tokens = tokens(source_patch_text(state))
    impact = []
    for item in path_prefilter(state, args.prefilter_files):
        text = target_file_text(state, runner, args, item["path"], content_cache)
        if not text:
            continue
        target_tokens = tokens(text[: args.max_file_chars])
        reference_score = jaccard(source_symbols, target_tokens)
        intent_score = jaccard(patch_tokens, target_tokens)
        if reference_score <= 0.015 and intent_score <= 0.035:
            continue
        score = 0.58 * reference_score + 0.27 * intent_score + 0.15 * float(item["score"])
        impact.append(
            {
                **item,
                "score": round(score, 6),
                "action": "FindImpactNeighbor",
                "evidence": {
                    "changed_identifier_reference": round(reference_score, 6),
                    "patch_intent_overlap": round(intent_score, 6),
                    "path_prior": round(float(item["score"]), 6),
                },
            }
        )
    impact.sort(key=lambda item: (-float(item["score"]), item["path"]))
    impact_rank = {item["path"]: rank for rank, item in enumerate(impact, 1)}
    by_path = {item["path"]: dict(item) for item in base}
    for item in impact[: args.candidate_files * 2]:
        by_path.setdefault(item["path"], dict(item))
    source_file = state["source_changed_paths"][0]
    ranked = []
    for path, item in by_path.items():
        # Reciprocal-rank fusion keeps the reliable PropQA head while allowing
        # strong reference evidence to promote impact-only neighbors.
        score = 0.74 / (2 + base_rank.get(path, args.candidate_files + 5))
        score += 0.26 / (2 + impact_rank.get(path, args.candidate_files * 2 + 5))
        if path == source_file:
            score += 0.30
        out = dict(item)
        out["fusion_score"] = round(score, 6)
        out["impact_rank"] = impact_rank.get(path)
        ranked.append(out)
    ranked.sort(key=lambda item: (-float(item["fusion_score"]), item["path"]))
    return ranked[: args.candidate_files]


def nicad_files(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace, content_cache: dict[tuple[str, str, str], str]) -> list[dict[str, Any]]:
    src_norm = normalize_code(source_patch_text(state))
    candidates = path_prefilter(state, args.prefilter_files)
    ranked = []
    for item in candidates:
        text = target_file_text(state, runner, args, item["path"], content_cache)
        clone_score = token_similarity(src_norm, normalize_code(text[: args.max_file_chars])) if text else 0.0
        score = 0.72 * clone_score + 0.28 * item["score"]
        ranked.append({**item, "score": round(score, 6), "baseline": "Open-NiCad"})
    return sorted(ranked, key=lambda x: (-x["score"], x["path"]))[: args.candidate_files]


def code2vec_files(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace, content_cache: dict[tuple[str, str, str], str]) -> list[dict[str, Any]]:
    source_file = state["source_changed_paths"][0]
    source_text = source_file_query_text(state, runner, args, source_file)
    source_token_set = tokens(source_text) | changed_symbol_terms(state)
    symbols = changed_symbol_terms(state)
    ranked = []
    for item in path_prefilter(state, args.prefilter_files):
        target_text = target_file_text(state, runner, args, item["path"], content_cache)
        target_token_set = tokens(target_text) | path_tokens(item["path"])
        content_score = jaccard(source_token_set, target_token_set)
        symbol_path_score = jaccard(symbols | source_terms(state), path_tokens(item["path"]))
        score = 0.62 * content_score + 0.23 * symbol_path_score + 0.15 * item["score"]
        ranked.append(
            {
                **item,
                "score": round(score, 6),
                "baseline": "code2vec-style",
                "evidence": {
                    "content_token_similarity": round(content_score, 6),
                    "symbol_path_similarity": round(symbol_path_score, 6),
                    "path_prior": round(float(item["score"]), 6),
                },
            }
        )
    return sorted(ranked, key=lambda x: (-x["score"], x["path"]))[: args.candidate_files]


def metrics_for_records(records: list[dict[str, Any]], predictions: dict[tuple[int, str, str], list[str]], methods: list[str], ks: list[int]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        ranks = []
        for rec in records:
            key = (rec["pair_index"], rec["source_file"], method)
            pred = predictions.get(key, [])
            target = rec["target_file"]
            rank = next((idx for idx, path in enumerate(pred, start=1) if path == target), None)
            ranks.append(rank)
        for k in ks:
            hit = sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks) if ranks else 0.0
            rows.append({"method": method, "metric": f"Hit@{k}", "value": hit, "file_pairs": len(ranks)})
        mrr = sum((1.0 / rank) if rank else 0.0 for rank in ranks) / len(ranks) if ranks else 0.0
        mean_rank = sum((rank if rank else args.candidate_files + 1) for rank in ranks) / len(ranks) if ranks else 0.0
        rows.append({"method": method, "metric": "MRR@5", "value": sum((1.0 / rank) if rank and rank <= 5 else 0.0 for rank in ranks) / len(ranks) if ranks else 0.0, "file_pairs": len(ranks)})
        rows.append({"method": method, "metric": "MeanRank", "value": mean_rank, "file_pairs": len(ranks)})
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    by_method: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][row["metric"]] = row["value"]
        by_method[row["method"]]["file_pairs"] = row["file_pairs"]
    lines = ["| Method | File pairs | Hit@1 | Hit@3 | Hit@5 | MRR@5 | MeanRank |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ["PropQA", "PropQA-Actions", "PropQA-Impact", "Open-NiCad", "code2vec-style"]:
        item = by_method.get(method, {})
        lines.append(
            f"| {method} | {int(item.get('file_pairs', 0))} | "
            f"{item.get('Hit@1', 0):.3f} | {item.get('Hit@3', 0):.3f} | {item.get('Hit@5', 0):.3f} | "
            f"{item.get('MRR@5', 0):.3f} | {item.get('MeanRank', 0):.2f} |"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"), sleep_seconds=args.sleep)
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=args.candidate_files, top_statements=10)
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    content_cache: dict[tuple[str, str, str], str] = {}
    file_pair_records = []
    predictions: dict[tuple[int, str, str], list[str]] = {}
    query_details = []

    for index, pair in enumerate(pairs, start=1):
        setup = runner._prepare_pair(pair)
        if setup.get("status") != "ok":
            print(f"[{index}/{len(pairs)}] skip {setup.get('status')}", flush=True)
            continue
        setup = normalize_target_oracle_to_dataset_files(setup)
        pair_records = build_file_pair_records(index, pair, setup)
        file_pair_records.extend(pair_records)
        source_files = sorted({rec["source_file"] for rec in pair_records})
        for source_file in source_files:
            state = restrict_state_to_source_file(setup, source_file)
            methods = {
                "PropQA": graphqa_files(state, args),
                "PropQA-Actions": graphqa_action_files(state, args),
                "PropQA-Impact": graphqa_impact_files(state, runner, args, content_cache),
                "Open-NiCad": nicad_files(state, runner, args, content_cache),
                "code2vec-style": code2vec_files(state, runner, args, content_cache),
            }
            for method, ranked in methods.items():
                predictions[(index, source_file, method)] = [item["path"] for item in ranked]
            query_details.append(
                {
                    "pair_index": index,
                    "source_file": source_file,
                    "target_gt_files": [rec["target_file"] for rec in pair_records if rec["source_file"] == source_file],
                    "predictions": {method: ranked[: args.candidate_files] for method, ranked in methods.items()},
                }
            )
        print(f"[{index}/{len(pairs)}] file_pairs={len(pair_records)} source_queries={len(source_files)}", flush=True)

    methods = ["PropQA", "PropQA-Actions", "PropQA-Impact", "Open-NiCad", "code2vec-style"]
    rows = metrics_for_records(file_pair_records, predictions, methods, args.eval_ks)
    return {
        "summary": {
            "input": args.input,
            "pairs": len(pairs),
            "file_pairs": len(file_pair_records),
            "source_file_queries": len(query_details),
            "candidate_files": args.candidate_files,
            "mapping": "target-covered best source plus source-covered best target, scored by path and patch similarity",
        },
        "rows": rows,
        "file_pair_records": file_pair_records,
        "query_details": query_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/URL_Results_detection_primary_v4.json")
    parser.add_argument("--output", default="outputs/file_pair_detection_graphqa_baselines.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--local-git-repo-cache", default=".cache/git_repos")
    parser.add_argument("--prefilter-files", type=int, default=40)
    parser.add_argument("--candidate-files", type=int, default=20)
    parser.add_argument("--max-file-chars", type=int, default=12000)
    parser.add_argument("--eval-ks", type=int, nargs="*", default=[1, 3, 5])
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--git-show-timeout", type=int, default=8)
    parser.add_argument("--allow-github-file-fetch", action="store_true")
    global args
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_out = out.with_name("file_pair_detection_dataset.json")
    dataset_out.write_text(json.dumps(payload["file_pair_records"], ensure_ascii=False, indent=2), encoding="utf-8")
    md = (
        "# File-to-File Detection Experiment\n\n"
        + f"- Original pairs: {payload['summary']['pairs']}\n"
        + f"- File-pair records: {payload['summary']['file_pairs']}\n"
        + f"- Source-file queries: {payload['summary']['source_file_queries']}\n"
        + f"- Mapping: {payload['summary']['mapping']}\n\n"
        + format_table(payload["rows"])
        + "\n"
    )
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    print(f"wrote {dataset_out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
