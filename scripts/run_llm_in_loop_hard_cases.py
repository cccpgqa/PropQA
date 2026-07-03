#!/usr/bin/env python3
"""Run a controlled LLM-in-the-loop QA agent on hard propagation cases.

The LLM participates in planning and final answer selection, but it is not
allowed to invent paths or lines. All executable actions are still backed by
the deterministic graph/AST tools from `run_qa_mcts_small_sample.py`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from propagation_detector import GitHubClient, line_hits_ranges
from run_qa_mcts_small_sample import (
    LLMClient,
    PropagationQAMCTS,
    extract_json_object,
    load_dotenv,
)


DEFAULT_HARD = "4,16,83,93,126,127,129,130,132,133,135,171,196"


def short_stmt(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": item.get("path"),
        "line": item.get("line"),
        "score": item.get("score"),
        "target_symbol": item.get("target_symbol"),
        "patch_pattern": item.get("patch_pattern"),
        "line_text": item.get("line_text", "")[:180],
        "source_path": item.get("source_path"),
        "source_text": item.get("source_text", "")[:180],
    }


def safe_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip().isdigit()]


class LoopLLM(LLMClient):
    def complete_json_with_system(self, system: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            self.last_error = "disabled"
            return None
        import urllib.error
        import urllib.request

        self.last_error = ""
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 4000,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = None
        errors = []
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                errors.append(f"HTTPError {exc.code}")
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if data is None:
            self.last_error = " | ".join(errors[-3:])
            return None
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        parsed = extract_json_object(content)
        if parsed is None:
            self.last_error = f"unparseable content: {content[:200]}"
        return parsed


def prepare_state(runner: PropagationQAMCTS, pair: dict[str, Any]) -> dict[str, Any]:
    setup = runner._prepare_pair(pair)
    if setup.get("status") != "ok":
        return setup
    return {
        **setup,
        "source_summary": None,
        "candidate_files": [],
        "mapped_statements": [],
        "mapped_files_attempted": [],
        "viewed_spans": [],
        "finished": False,
    }


def execute_and_update(runner: PropagationQAMCTS, state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    obs = runner._execute(state, action)
    runner._update_state(state, action, obs)
    return obs


def llm_choose_files(llm: LoopLLM, state: dict[str, Any], top_n: int, max_inspect: int) -> tuple[list[str], dict[str, Any] | None, str]:
    payload = {
        "task": "Choose target repository files to inspect for cross-repository change propagation. Select only paths from candidate_target_files.",
        "source": {
            "repo": state["propagator"].repo,
            "title": state["propagator"].title,
            "changed_files": state["source_changed_paths"][:20],
            "patch_statements": (state.get("source_summary") or {}).get("statements", [])[:18],
        },
        "target": {
            "repo": state["target"].repo,
            "title": state["target"].title,
            "candidate_target_files": state["candidate_files"][:top_n],
        },
        "output_schema": {
            "inspect_files": ["path"],
            "likely_files": [{"path": "path", "confidence": 0.0}],
        },
        "constraints": [
            f"Return at most {max_inspect} inspect_files.",
            "Every returned path must exactly match one provided candidate path.",
            "Return JSON only.",
        ],
    }
    system = (
        "You are an LLM planner inside a graph-guided code propagation QA agent. "
        "Choose useful tool actions, but never invent paths."
    )
    result = llm.complete_json_with_system(system, payload)
    allowed = {item["path"] for item in state["candidate_files"][:top_n]}
    chosen: list[str] = []
    if result:
        for path in result.get("inspect_files", []):
            if isinstance(path, str) and path in allowed and path not in chosen:
                chosen.append(path)
        for item in result.get("likely_files", []):
            path = item.get("path") if isinstance(item, dict) else None
            if path in allowed and path not in chosen:
                chosen.append(path)
    if not chosen:
        chosen = [item["path"] for item in state["candidate_files"][:max_inspect]]
    return chosen[:max_inspect], result, llm.last_error


def llm_final_answer(llm: LoopLLM, state: dict[str, Any], top_files: int, top_statements: int) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    file_candidates = state["candidate_files"][:top_files]
    stmt_candidates = [short_stmt(item) for item in state["mapped_statements"][: max(top_statements, 24)]]
    payload = {
        "task": "Return final propagation localization answer. Rerank only provided candidates; do not invent paths or lines.",
        "source": {
            "repo": state["propagator"].repo,
            "title": state["propagator"].title,
            "changed_files": state["source_changed_paths"][:20],
            "patch_statements": (state.get("source_summary") or {}).get("statements", [])[:18],
        },
        "target": {
            "repo": state["target"].repo,
            "title": state["target"].title,
        },
        "candidate_target_files": file_candidates,
        "candidate_target_statements": stmt_candidates,
        "output_schema": {
            "target_files": [{"path": "path", "confidence": 0.0}],
            "target_statements": [{"path": "path", "line": 1, "confidence": 0.0}],
        },
        "constraints": [
            "Every target file path must be copied from candidate_target_files.",
            "Every target statement path+line must be copied from candidate_target_statements.",
            "Return JSON only.",
        ],
    }
    system = (
        "You are the reasoning component of a DeepRepoQA-style propagation localization agent. "
        "Use source patch semantics, file mapping, AST/symbol hints, and statement candidates to answer."
    )
    result = llm.complete_json_with_system(system, payload)
    file_by_path = {item["path"]: item for item in file_candidates}
    stmt_by_key = {(item["path"], int(item["line"])): item for item in state["mapped_statements"]}

    target_files = []
    if result:
        for item in result.get("target_files", []):
            path = item.get("path") if isinstance(item, dict) else None
            if path in file_by_path and path not in {x["path"] for x in target_files}:
                target_files.append(
                    {
                        "path": path,
                        "confidence": float(item.get("confidence", file_by_path[path].get("score", 0.0)) or 0.0),
                    }
                )
        for item in result.get("target_statements", []):
            path = item.get("path") if isinstance(item, dict) else None
            try:
                line = int(item.get("line"))
            except Exception:
                continue
            if (path, line) in stmt_by_key and path not in {x["path"] for x in target_files}:
                target_files.append({"path": path, "confidence": 0.5})
    for item in file_candidates:
        if len(target_files) >= top_files:
            break
        if item["path"] not in {x["path"] for x in target_files}:
            target_files.append({"path": item["path"], "confidence": float(item.get("score", 0.0))})

    target_statements = []
    if result:
        for item in result.get("target_statements", []):
            path = item.get("path") if isinstance(item, dict) else None
            try:
                line = int(item.get("line"))
            except Exception:
                continue
            if (path, line) in stmt_by_key and (path, line) not in {(x["path"], x["line"]) for x in target_statements}:
                old = dict(stmt_by_key[(path, line)])
                old["llm_confidence"] = float(item.get("confidence", old.get("score", 0.0)) or 0.0)
                target_statements.append(old)
    for item in state["mapped_statements"]:
        if len(target_statements) >= top_statements:
            break
        key = (item["path"], item["line"])
        if key not in {(x["path"], x["line"]) for x in target_statements}:
            target_statements.append(item)

    prediction = {
        "propagation_likely": bool(target_files),
        "target_files": target_files[:top_files],
        "target_statements": target_statements[:top_statements],
        "llm_used": bool(result),
        "llm_error": None if result else llm.last_error,
    }
    return prediction, result, llm.last_error


def evaluate_prediction(setup_or_state: dict[str, Any], prediction: dict[str, Any], tolerance: int = 5) -> dict[str, Any]:
    gt_files = set(setup_or_state["target_changed_paths"])
    predicted_files = [item["path"] for item in prediction["target_files"]]
    file_hit_paths = [path for path in predicted_files if path in gt_files]
    statement_hits = []
    for item in prediction["target_statements"]:
        if line_hits_ranges(item["path"], int(item["line"]), setup_or_state["target_ranges"], tolerance=tolerance):
            statement_hits.append(item)
    return {
        "file_topk_hit": bool(file_hit_paths),
        "file_hit_paths": file_hit_paths,
        "statement_topk_hit": bool(statement_hits),
        "statement_hits": statement_hits[:5],
    }


def run_one(pair: dict[str, Any], index: int, client: GitHubClient, llm: LoopLLM, args: argparse.Namespace) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=args.top_files, top_statements=args.candidate_statements)
    state = prepare_state(runner, pair)
    if state.get("status") and state.get("status") != "ok":
        return {"index": index, **state}

    trajectory = []
    obs = execute_and_update(runner, state, {"type": "InspectSourcePatch"})
    trajectory.append({"action": "InspectSourcePatch", "observation": obs})
    obs = execute_and_update(runner, state, {"type": "FindSimilarFiles", "top_k": args.candidate_files})
    trajectory.append({"action": "FindSimilarFiles", "observation": {"candidate_files": obs.get("candidate_files", [])[: args.top_files]}})

    chosen_files, plan_json, plan_error = llm_choose_files(llm, state, args.candidate_files, args.inspect_files)
    mapped_observations = []
    by_path = {item["path"]: item for item in state["candidate_files"]}
    for path in chosen_files:
        item = by_path.get(path, {})
        obs = execute_and_update(
            runner,
            state,
            {"type": "MapPatchToTarget", "path": path, "matched_source_path": item.get("matched_source_path")},
        )
        mapped_observations.append({"path": path, "mapped_statement_count": len(obs.get("mapped_statements", []))})
    trajectory.append({"action": "LLMPlan+MapPatchToTarget", "llm_plan": plan_json, "llm_error": plan_error, "mapped": mapped_observations})

    prediction, final_json, final_error = llm_final_answer(llm, state, args.top_files, args.top_statements)
    metrics = evaluate_prediction(state, prediction)
    return {
        "index": index,
        "status": "ok",
        "propagator": {
            "repo": state["propagator"].repo,
            "kind": state["propagator"].kind,
            "number": state["propagator"].number,
            "title": state["propagator"].title,
            "file_names": state["source_changed_paths"],
        },
        "target": {
            "repo": state["target"].repo,
            "kind": state["target"].kind,
            "number": state["target"].number,
            "title": state["target"].title,
            "file_names": state["target_changed_paths"],
        },
        "trajectory": trajectory,
        "prediction": prediction,
        "ground_truth": {
            "target_changed_files": state["target_changed_paths"],
            "target_changed_old_line_ranges": {
                path: [{"start": start, "end": end} for start, end in ranges]
                for path, ranges in state["target_ranges"].items()
            },
        },
        "metrics": metrics,
        "llm_final": final_json,
        "llm_final_error": final_error,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "evaluated_pairs": len(ok),
        "skipped_pairs": len(results) - len(ok),
        "file_topk_hits": sum(1 for r in ok if r.get("metrics", {}).get("file_topk_hit")),
        "file_topk_hit_rate": round(sum(1 for r in ok if r.get("metrics", {}).get("file_topk_hit")) / len(ok), 4) if ok else None,
        "statement_topk_hits": sum(1 for r in ok if r.get("metrics", {}).get("statement_topk_hit")),
        "statement_topk_hit_rate": round(sum(1 for r in ok if r.get("metrics", {}).get("statement_topk_hit")) / len(ok), 4) if ok else None,
        "llm_used": sum(1 for r in ok if r.get("prediction", {}).get("llm_used")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/URL_Results_detection_subset.json")
    parser.add_argument("--output", default="outputs/qa_mcts_llm_in_loop_hard15.json")
    parser.add_argument("--indices", default=DEFAULT_HARD)
    parser.add_argument("--candidate-files", type=int, default=20)
    parser.add_argument("--inspect-files", type=int, default=8)
    parser.add_argument("--candidate-statements", type=int, default=24)
    parser.add_argument("--top-files", type=int, default=8)
    parser.add_argument("--top-statements", type=int, default=8)
    parser.add_argument("--llm-timeout", type=int, default=180)
    args = parser.parse_args()

    load_dotenv()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    indices = safe_ints(args.indices)
    client = GitHubClient(Path(".cache/github"), token=os.environ.get("GITHUB_TOKEN"))
    llm = LoopLLM(timeout=args.llm_timeout)
    results = []
    for offset, index in enumerate(indices, start=1):
        try:
            result = run_one(pairs[index - 1], index, client, llm, args)
        except Exception as exc:
            result = {"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        payload = {"summary": summarize(results), "results": results}
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{offset}/{len(indices)}; index={index}] {result['status']} {result.get('reason', '')}", flush=True)
    payload = {"summary": summarize(results), "results": results}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
