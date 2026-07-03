#!/usr/bin/env python3
"""Run a small graph-style QA/MCTS propagation detection experiment.

This is an executable prototype of the agent design in
`docs/propagation_qa_agent_design.md`. It uses deterministic agents so the
pipeline can be evaluated without an LLM key:

- Perception: summarizes current evidence.
- Planning: proposes propagation-localization actions.
- Execution: runs actions against GitHub-backed graph/code data.
- Evaluation: assigns values for MCTS selection/backpropagation.

The target after-fix patch is used only for evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from propagation_detector import (
    GitHubClient,
    changed_old_line_ranges,
    iter_patch_statements,
    meaningful_statement,
    parsed_sides,
    path_score,
    remote_changed_files,
    resolve_side,
    token_similarity,
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
GO_SYMBOL_TYPES = {
    "function_declaration",
    "method_declaration",
    "type_declaration",
    "const_declaration",
    "var_declaration",
}
BOILERPLATE_PREFIXES = (
    "package ",
    "import ",
    "//",
    "/*",
    "*",
)


@dataclasses.dataclass
class SearchNode:
    node_id: int
    parent_id: int | None
    action: dict[str, Any]
    observation: dict[str, Any]
    value: float = 0.0
    visits: int = 0
    children: list[int] = dataclasses.field(default_factory=list)


class PropagationQAMCTS:
    def __init__(
        self,
        client: GitHubClient,
        max_nodes: int = 15,
        exploration: float = 1.2,
        top_files: int = 8,
        top_statements: int = 10,
        llm: "LLMClient | None" = None,
        llm_hard_only: bool = False,
    ) -> None:
        self.client = client
        self.max_nodes = max_nodes
        self.exploration = exploration
        self.top_files = top_files
        self.top_statements = top_statements
        self.nodes: dict[int, SearchNode] = {}
        self.next_node_id = 0
        self.ast = ASTAnalyzer()
        self.llm = llm
        self.llm_hard_only = llm_hard_only

    def run_pair(self, pair: dict[str, Any], index: int) -> dict[str, Any]:
        setup = self._prepare_pair(pair)
        if setup.get("status") != "ok":
            return {"index": index, **setup}

        root = self._new_node(None, {"type": "Root"}, {"summary": "start"})
        state = {
            **setup,
            "source_summary": None,
            "candidate_files": [],
            "mapped_statements": [],
            "mapped_files_attempted": [],
            "viewed_spans": [],
            "finished": False,
        }

        for _ in range(self.max_nodes):
            leaf = self._select(root.node_id)
            perception = self._perceive(state, leaf)
            actions = self._plan(state, perception)
            if not actions:
                break
            for action in actions[:3]:
                observation = self._execute(state, action)
                child = self._new_node(leaf.node_id, action, observation)
                leaf.children.append(child.node_id)
                value = self._evaluate(state, action, observation)
                self._backpropagate(child.node_id, value)
                self._update_state(state, action, observation)
                if action["type"] == "Finish":
                    state["finished"] = True
                    break
            if state["finished"]:
                break

        prediction = self._finish_prediction(state)
        if self.llm and (not self.llm_hard_only or self._llm_needed(state, prediction)):
            prediction = self._llm_rerank_prediction(state, prediction)
        metrics = self._evaluate_prediction(setup, prediction)
        hardness = self._hardness(setup, prediction, metrics)
        return {
            "index": index,
            "status": "ok",
            "propagator": dataclasses.asdict(setup["propagator"]),
            "target": dataclasses.asdict(setup["target"]),
            "trajectory": [dataclasses.asdict(self.nodes[node_id]) for node_id in sorted(self.nodes)],
            "prediction": prediction,
            "ground_truth": {
                "target_changed_files": setup["target_changed_paths"],
                "target_changed_old_line_ranges": {
                    path: [{"start": start, "end": end} for start, end in ranges]
                    for path, ranges in setup["target_ranges"].items()
                },
            },
            "metrics": metrics,
            "hardness": hardness,
        }

    def _prepare_pair(self, pair: dict[str, Any]) -> dict[str, Any]:
        sides = parsed_sides(pair)
        if sides is None:
            return {"status": "skipped", "reason": "missing side metadata"}
        source_side, infestor_side = (resolve_side(self.client, side) for side in sides)
        if not (source_side.base_sha and source_side.merged_at and infestor_side.base_sha and infestor_side.merged_at):
            return {"status": "skipped", "reason": "missing base/date metadata"}
        propagator, target = (source_side, infestor_side) if source_side.merged_at <= infestor_side.merged_at else (infestor_side, source_side)

        source_files = remote_changed_files(self.client, propagator)
        target_files = remote_changed_files(self.client, target)
        source_files = prefer_side_files(source_files, propagator.file_names)
        source_changed_paths = [f.get("filename") for f in source_files if f.get("filename")]
        target_changed_paths = [f.get("filename") for f in target_files if f.get("filename")]
        target_tree = self.client.tree_paths(target.repo, target.base_sha)
        return {
            "status": "ok",
            "propagator": propagator,
            "target": target,
            "source_files": source_files,
            "target_files": target_files,
            "source_changed_paths": source_changed_paths,
            "target_changed_paths": target_changed_paths,
            "source_statements": iter_patch_statements(source_files, include_context=True),
            "target_tree": target_tree,
            "target_ranges": changed_old_line_ranges(target_files),
            "source_symbol_cache": {},
            "target_symbol_cache": {},
        }

    def _new_node(self, parent_id: int | None, action: dict[str, Any], observation: dict[str, Any]) -> SearchNode:
        node = SearchNode(self.next_node_id, parent_id, action, observation)
        self.nodes[node.node_id] = node
        self.next_node_id += 1
        return node

    def _select(self, root_id: int) -> SearchNode:
        node = self.nodes[root_id]
        while node.children:
            node = max((self.nodes[c] for c in node.children), key=lambda child: self._uct(node, child))
        return node

    def _uct(self, parent: SearchNode, child: SearchNode) -> float:
        if child.visits == 0:
            return float("inf")
        return child.value / child.visits + self.exploration * math.sqrt(math.log(parent.visits + 1) / child.visits)

    def _perceive(self, state: dict[str, Any], leaf: SearchNode) -> dict[str, Any]:
        return {
            "has_source_summary": state["source_summary"] is not None,
            "candidate_file_count": len(state["candidate_files"]),
            "mapped_statement_count": len(state["mapped_statements"]),
            "best_file": state["candidate_files"][0]["path"] if state["candidate_files"] else None,
            "best_statement": state["mapped_statements"][0] if state["mapped_statements"] else None,
            "leaf_action": leaf.action,
        }

    def _plan(self, state: dict[str, Any], perception: dict[str, Any]) -> list[dict[str, Any]]:
        if state["source_summary"] is None:
            return [{"type": "InspectSourcePatch"}]
        if not state["candidate_files"]:
            return [{"type": "FindSimilarFiles", "top_k": self.top_files}]
        mapped_paths = set(state["mapped_files_attempted"])
        unmapped = [item for item in state["candidate_files"] if item["path"] not in mapped_paths]
        if unmapped:
            return [
                {
                    "type": "MapPatchToTarget",
                    "path": item["path"],
                    "matched_source_path": item.get("matched_source_path"),
                }
                for item in unmapped[:3]
            ]
        unviewed = [item for item in state["mapped_statements"][:3] if (item["path"], item["line"]) not in state["viewed_spans"]]
        if unviewed:
            return [{"type": "ViewCode", "path": item["path"], "line": item["line"], "radius": 4} for item in unviewed[:2]]
        return [{"type": "Finish"}]

    def _execute(self, state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        if action["type"] == "InspectSourcePatch":
            statements = [
                {
                    "path": stmt.path,
                    "line": stmt.line_no,
                    "kind": stmt.kind,
                    "text": stmt.text[:240],
                }
                for stmt in state["source_statements"][:80]
            ]
            return {"source_changed_files": state["source_changed_paths"], "statements": statements}

        if action["type"] == "FindSimilarFiles":
            candidates: dict[str, dict[str, Any]] = {}
            for source_path in state["source_changed_paths"]:
                for target_path in state["target_tree"]:
                    score = path_score(source_path, target_path)
                    old = candidates.get(target_path)
                    if old is None or score > old["score"]:
                        candidates[target_path] = {
                            "path": target_path,
                            "score": round(score, 6),
                            "matched_source_path": source_path,
                        }
            ranked = sorted(candidates.values(), key=lambda item: (-item["score"], item["path"]))[: action["top_k"]]
            return {"candidate_files": ranked}

        if action["type"] == "MapPatchToTarget":
            mapped = self._map_patch_to_target(state, action["path"], action.get("matched_source_path"))
            return {"path": action["path"], "mapped_statements": mapped}

        if action["type"] == "ViewCode":
            text = self.client.file_at(state["target"].repo, state["target"].base_sha, action["path"]) or ""
            lines = text.splitlines()
            start = max(1, int(action["line"]) - int(action["radius"]))
            end = min(len(lines), int(action["line"]) + int(action["radius"]))
            snippet = "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end + 1))
            return {"path": action["path"], "start": start, "end": end, "snippet": snippet}

        if action["type"] == "Finish":
            return {"answer_ready": True}

        return {"error": f"unknown action {action['type']}"}

    def _map_patch_to_target(self, state: dict[str, Any], target_path: str, matched_source_path: str | None) -> list[dict[str, Any]]:
        content = self.client.file_at(state["target"].repo, state["target"].base_sha, target_path)
        if not content:
            return []
        source_statements = [
            stmt
            for stmt in state["source_statements"]
            if meaningful_statement(stmt.text)
            and (not matched_source_path or stmt.path == matched_source_path)
        ]
        if not source_statements:
            source_basename = (matched_source_path or "").rsplit("/", 1)[-1]
            source_statements = [
                stmt
                for stmt in state["source_statements"]
                if meaningful_statement(stmt.text) and stmt.path.rsplit("/", 1)[-1] == source_basename
            ]
        if not source_statements:
            source_statements = [
                stmt for stmt in state["source_statements"] if meaningful_statement(stmt.text)
            ][:80]
        if not source_statements:
            return []
        source_symbols = self._source_symbols_for_patch(state, matched_source_path, source_statements)
        target_symbols = self._symbols_for_file(
            state,
            role="target",
            repo=state["target"].repo,
            sha=state["target"].base_sha,
            path=target_path,
        )
        pattern = classify_patch_pattern(source_statements)
        symbol_candidates = rank_target_symbols(source_symbols, target_symbols, pattern)
        allowed_ranges = [(sym["start_line"], sym["end_line"], sym) for sym in symbol_candidates[:4]]
        if not allowed_ranges:
            allowed_ranges = [(1, len(content.splitlines()), None)]

        ranked = []
        lines = content.splitlines()
        for line_no, line in enumerate(lines, start=1):
            symbol_boost = 0.0
            symbol_name = None
            if allowed_ranges:
                in_allowed = False
                for start, end, sym in allowed_ranges:
                    if start <= line_no <= end:
                        in_allowed = True
                        if sym:
                            symbol_boost = float(sym.get("match_score", 0.0))
                            symbol_name = sym.get("name")
                        break
                if not in_allowed:
                    continue
            stripped = line.strip()
            if not meaningful_statement(stripped):
                continue
            if is_boilerplate_statement(stripped):
                continue
            best = max(source_statements, key=lambda stmt: token_similarity(stmt.text, stripped))
            sim = token_similarity(best.text, stripped)
            pattern_score = pattern_line_score(pattern, stripped)
            score = 0.58 * sim + 0.25 * symbol_boost + 0.17 * pattern_score
            ranked.append(
                {
                    "path": target_path,
                    "line": line_no,
                    "score": round(score, 6),
                    "line_similarity": round(sim, 6),
                    "symbol_score": round(symbol_boost, 6),
                    "pattern_score": round(pattern_score, 6),
                    "target_symbol": symbol_name,
                    "patch_pattern": pattern,
                    "line_text": stripped[:240],
                    "source_path": best.path,
                    "source_line": best.line_no,
                    "source_kind": best.kind,
                    "source_text": best.text[:240],
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["line"]))
        return ranked[: self.top_statements]

    def _source_symbols_for_patch(
        self,
        state: dict[str, Any],
        matched_source_path: str | None,
        source_statements: list[Any],
    ) -> list[dict[str, Any]]:
        paths = []
        if matched_source_path:
            paths.append(matched_source_path)
        paths.extend(stmt.path for stmt in source_statements)
        out = []
        seen = set()
        for path in paths:
            if not path or path in seen:
                continue
            seen.add(path)
            symbols = self._symbols_for_file(
                state,
                role="source",
                repo=state["propagator"].repo,
                sha=state["propagator"].base_sha,
                path=path,
            )
            hit_lines = {stmt.line_no for stmt in source_statements if stmt.path == path}
            hit_symbols = [
                sym
                for sym in symbols
                if any(sym["start_line"] <= line <= sym["end_line"] for line in hit_lines)
            ]
            out.extend(hit_symbols or symbols[:4])
        return out[:8]

    def _symbols_for_file(
        self,
        state: dict[str, Any],
        role: str,
        repo: str,
        sha: str,
        path: str,
    ) -> list[dict[str, Any]]:
        cache_name = "source_symbol_cache" if role == "source" else "target_symbol_cache"
        cache = state[cache_name]
        key = (repo, sha, path)
        if key in cache:
            return cache[key]
        content = self.client.file_at(repo, sha, path)
        symbols = self.ast.extract_symbols(path, content or "")
        cache[key] = symbols
        return symbols

    def _evaluate(self, state: dict[str, Any], action: dict[str, Any], observation: dict[str, Any]) -> float:
        if action["type"] == "InspectSourcePatch":
            return 55.0 if observation.get("statements") else 25.0
        if action["type"] == "FindSimilarFiles":
            best = observation.get("candidate_files", [{}])[0].get("score", 0)
            return 35.0 + 45.0 * float(best)
        if action["type"] == "MapPatchToTarget":
            mapped = observation.get("mapped_statements") or []
            if not mapped:
                return 20.0
            return 45.0 + 50.0 * float(mapped[0]["score"])
        if action["type"] == "ViewCode":
            return 80.0 if observation.get("snippet") else 20.0
        if action["type"] == "Finish":
            return 90.0 if state["mapped_statements"] else 30.0
        return 0.0

    def _backpropagate(self, node_id: int, value: float) -> None:
        while node_id is not None:
            node = self.nodes[node_id]
            node.visits += 1
            node.value += value
            node_id = node.parent_id

    def _update_state(self, state: dict[str, Any], action: dict[str, Any], observation: dict[str, Any]) -> None:
        if action["type"] == "InspectSourcePatch":
            state["source_summary"] = observation
        elif action["type"] == "FindSimilarFiles":
            state["candidate_files"] = observation.get("candidate_files") or []
        elif action["type"] == "MapPatchToTarget":
            if action["path"] not in state["mapped_files_attempted"]:
                state["mapped_files_attempted"].append(action["path"])
            state["mapped_statements"].extend(observation.get("mapped_statements") or [])
            state["mapped_statements"].sort(key=lambda item: (-item["score"], item["path"], item["line"]))
            state["mapped_statements"] = dedupe_statement_predictions(state["mapped_statements"])[: self.top_statements]
        elif action["type"] == "ViewCode":
            state["viewed_spans"].append((action["path"], action["line"]))

    def _finish_prediction(self, state: dict[str, Any]) -> dict[str, Any]:
        file_scores: dict[str, float] = {}
        for item in state["candidate_files"]:
            file_scores[item["path"]] = max(file_scores.get(item["path"], 0.0), float(item["score"]))
        for item in state["mapped_statements"]:
            file_scores[item["path"]] = max(file_scores.get(item["path"], 0.0), float(item["score"]))
        target_files = [
            {"path": path, "confidence": round(score, 6)}
            for path, score in sorted(file_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_files]
        ]
        return {
            "propagation_likely": bool(target_files),
            "target_files": target_files,
            "target_statements": state["mapped_statements"][: self.top_statements],
        }

    def _evaluate_prediction(self, setup: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        gt_files = set(setup["target_changed_paths"])
        predicted_files = [item["path"] for item in prediction["target_files"]]
        file_hit = any(path in gt_files for path in predicted_files)
        statement_hits = []
        for item in prediction["target_statements"]:
            if line_hits_ranges(item["path"], int(item["line"]), setup["target_ranges"], tolerance=5):
                statement_hits.append(item)
        return {
            "file_topk_hit": file_hit,
            "file_hit_paths": [path for path in predicted_files if path in gt_files],
            "statement_topk_hit": bool(statement_hits),
            "statement_hits": statement_hits[:3],
        }

    def _llm_rerank_prediction(self, state: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        source_patch = (state.get("source_summary") or {}).get("statements", [])[:20]
        payload = {
            "task": "Rerank target files and statements for cross-repository propagation localization. Do not invent new files. Use only provided candidates.",
            "source_changed_files": state["source_changed_paths"][:20],
            "source_patch_statements": source_patch[:10],
            "candidate_target_files": prediction["target_files"][: self.top_files],
            "candidate_target_statements": prediction["target_statements"][: self.top_statements],
            "output_schema": {
                "target_files": [{"path": "string", "confidence": 0.0}],
                "target_statements": [{"path": "string", "line": 1, "confidence": 0.0}],
            },
        }
        result = self.llm.complete_json(payload)
        if not result:
            return {
                **prediction,
                "llm_attempted": True,
                "llm_used": False,
                "llm_error": getattr(self.llm, "last_error", "no JSON result"),
            }
        file_by_path = {item["path"]: item for item in prediction["target_files"]}
        stmt_by_key = {
            (item["path"], int(item["line"])): item for item in prediction["target_statements"]
        }
        reranked_files = []
        for item in result.get("target_files", []):
            path = item.get("path")
            if path not in file_by_path:
                continue
            old = dict(file_by_path[path])
            old["llm_confidence"] = safe_float(item.get("confidence"), old.get("confidence", 0.0))
            old["llm_reason"] = str(item.get("reason", ""))[:300]
            old["confidence"] = round(0.45 * float(old.get("confidence", 0.0)) + 0.55 * old["llm_confidence"], 6)
            reranked_files.append(old)
        for item in prediction["target_files"]:
            if item["path"] not in {x["path"] for x in reranked_files}:
                reranked_files.append(item)
        reranked_files.sort(key=lambda item: (-float(item.get("confidence", 0.0)), item["path"]))

        reranked_statements = []
        for item in result.get("target_statements", []):
            key = (item.get("path"), int(item.get("line", -1)))
            if key not in stmt_by_key:
                continue
            old = dict(stmt_by_key[key])
            old["llm_confidence"] = safe_float(item.get("confidence"), old.get("score", 0.0))
            old["llm_reason"] = str(item.get("reason", ""))[:300]
            old["score"] = round(0.45 * float(old.get("score", 0.0)) + 0.55 * old["llm_confidence"], 6)
            reranked_statements.append(old)
        seen = {(x["path"], int(x["line"])) for x in reranked_statements}
        for item in prediction["target_statements"]:
            key = (item["path"], int(item["line"]))
            if key not in seen:
                reranked_statements.append(item)
        reranked_statements.sort(key=lambda item: (-float(item.get("score", 0.0)), item["path"], int(item["line"])))
        return {
            **prediction,
            "target_files": reranked_files[: self.top_files],
            "target_statements": reranked_statements[: self.top_statements],
            "llm_attempted": True,
            "llm_used": True,
        }

    def _llm_needed(self, state: dict[str, Any], prediction: dict[str, Any]) -> bool:
        source_paths = set(state["source_changed_paths"])
        candidate_paths = [item["path"] for item in prediction["target_files"]]
        exact_candidate = bool(source_paths & set(candidate_paths))
        many_source_files = len(source_paths) > 8
        top_scores = [round(float(item.get("confidence", 0.0)), 3) for item in prediction["target_files"][:5]]
        tied_top = len(top_scores) >= 3 and len(set(top_scores[:3])) == 1
        no_statement = not prediction.get("target_statements")
        return many_source_files or tied_top or no_statement or not exact_candidate

    def _hardness(self, setup: dict[str, Any], prediction: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
        source_paths = set(setup["source_changed_paths"])
        target_paths = set(setup["target_changed_paths"])
        source_basenames = {p.rsplit("/", 1)[-1] for p in source_paths}
        target_basenames = {p.rsplit("/", 1)[-1] for p in target_paths}
        source_stmt_texts = normalize_statement_set([stmt.text for stmt in setup["source_statements"]])
        target_stmt_texts = normalize_statement_set(
            [stmt.text for stmt in iter_patch_statements(setup["target_files"], include_context=True)]
        )
        flags = {
            "path_not_exact": len(source_paths & target_paths) == 0,
            "basename_not_exact": len(source_basenames & target_basenames) == 0,
            "statement_text_not_exact": len(source_stmt_texts & target_stmt_texts) == 0,
            "many_source_files": len(source_paths) > 10,
            "many_target_files": len(target_paths) > 10,
            "file_worked_despite_path_change": metrics["file_topk_hit"] and len(source_paths & target_paths) == 0,
            "statement_worked_despite_text_change": metrics["statement_topk_hit"] and len(source_stmt_texts & target_stmt_texts) == 0,
        }
        return {
            **flags,
            "source_file_count": len(source_paths),
            "target_file_count": len(target_paths),
            "path_overlap_count": len(source_paths & target_paths),
            "basename_overlap_count": len(source_basenames & target_basenames),
            "statement_text_overlap_count": len(source_stmt_texts & target_stmt_texts),
        }


def dedupe_statement_predictions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = (item["path"], item["line"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def prefer_side_files(files: list[dict[str, Any]], preferred_paths: list[str]) -> list[dict[str, Any]]:
    preferred = {path for path in preferred_paths if path}
    if not preferred:
        return files
    filtered = [
        file_info
        for file_info in files
        if (file_info.get("filename") or file_info.get("previous_filename")) in preferred
    ]
    return filtered or files


class ASTAnalyzer:
    def __init__(self) -> None:
        self.enabled = False
        self.parser = None
        try:
            import tree_sitter_language_pack

            ts_cache = Path(".cache/tree-sitter-language-pack").resolve()
            ts_cache.mkdir(parents=True, exist_ok=True)
            tree_sitter_language_pack.configure(cache_dir=str(ts_cache))
            self.parser = tree_sitter_language_pack.get_parser("go")
            self.enabled = True
        except Exception:
            self.enabled = False

    def extract_symbols(self, path: str, content: str) -> list[dict[str, Any]]:
        if not content:
            return []
        if not self.enabled or not path.endswith(".go"):
            return fallback_go_symbols(content)
        try:
            tree = self.parser.parse(content.encode("utf-8"))
            root = tree.root_node
            symbols = []
            stack = list(root.children)
            while stack:
                node = stack.pop()
                if node.type in GO_SYMBOL_TYPES:
                    text = node.text.decode("utf-8", errors="replace")
                    symbols.append(
                        {
                            "name": extract_symbol_name(text, node.type),
                            "kind": node.type,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "text": text[:4000],
                            "tokens": sorted(set(TOKEN_RE.findall(text.lower()))),
                            "ast_types": ast_type_counts(node),
                        }
                    )
                stack.extend(node.children)
            return symbols
        except Exception:
            return fallback_go_symbols(content)


def ast_type_counts(node: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    stack = [node]
    while stack:
        current = stack.pop()
        counts[current.type] = counts.get(current.type, 0) + 1
        stack.extend(current.children)
    return counts


def fallback_go_symbols(content: str) -> list[dict[str, Any]]:
    symbols = []
    lines = content.splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not (stripped.startswith("func ") or stripped.startswith("type ") or stripped.startswith("var ") or stripped.startswith("const ")):
            continue
        end = min(len(lines), idx + 80)
        text = "\n".join(lines[idx - 1 : end])
        symbols.append(
            {
                "name": extract_symbol_name(stripped, "fallback"),
                "kind": "fallback",
                "start_line": idx,
                "end_line": end,
                "text": text[:4000],
                "tokens": sorted(set(TOKEN_RE.findall(text.lower()))),
                "ast_types": {},
            }
        )
    return symbols


def extract_symbol_name(text: str, kind: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    if kind == "method_declaration" or first.startswith("func ("):
        match = re.search(r"\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", first)
        if match:
            return match.group(1)
    if kind == "function_declaration" or first.startswith("func "):
        match = re.search(r"func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", first)
        if match:
            return match.group(1)
    match = re.search(r"(?:type|var|const)\s+([A-Za-z_][A-Za-z0-9_]*)", first)
    if match:
        return match.group(1)
    return first[:80]


def rank_target_symbols(
    source_symbols: list[dict[str, Any]],
    target_symbols: list[dict[str, Any]],
    pattern: str,
) -> list[dict[str, Any]]:
    if not target_symbols:
        return []
    if not source_symbols:
        ranked = []
        for sym in target_symbols:
            copy = dict(sym)
            copy["match_score"] = 0.35
            ranked.append(copy)
        return ranked
    ranked = []
    for target in target_symbols:
        best = 0.0
        for source in source_symbols:
            name_score = name_similarity(source.get("name", ""), target.get("name", ""))
            token_score = jaccard(set(source.get("tokens", [])), set(target.get("tokens", [])))
            ast_score = weighted_ast_similarity(source.get("ast_types", {}), target.get("ast_types", {}))
            pattern_score = pattern_symbol_score(pattern, target)
            score = 0.34 * name_score + 0.28 * token_score + 0.24 * ast_score + 0.14 * pattern_score
            best = max(best, score)
        copy = dict(target)
        copy["match_score"] = round(best, 6)
        ranked.append(copy)
    ranked.sort(key=lambda item: (-item["match_score"], item["start_line"]))
    return ranked


def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_parts = set(split_identifier(a))
    b_parts = set(split_identifier(b))
    return max(token_similarity(a, b), jaccard(a_parts, b_parts))


def split_identifier(name: str) -> list[str]:
    chunks = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return [part for part in re.split(r"[_\W]+", chunks) if part]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def weighted_ast_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    intersection = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return intersection / union if union else 0.0


def classify_patch_pattern(source_statements: list[Any]) -> str:
    text = "\n".join(stmt.text.lower() for stmt in source_statements[:80])
    if any(word in text for word in ["limit", "max", "cap", "bound", "too many", "percentile"]):
        return "bounds_or_limit"
    if any(word in text for word in ["lock", "unlock", "mutex", "atomic", "race", "deadlock", "channel", "close"]):
        return "concurrency"
    if any(word in text for word in ["err", "error", "return nil", "panic", "recover"]):
        return "error_handling"
    if any(word in text for word in ["cache", "freezer", "database", "sync", "flush"]):
        return "storage_cache"
    if any(word in text for word in ["test", "assert", "require.", "t."]):
        return "test_change"
    return "generic"


def pattern_symbol_score(pattern: str, symbol: dict[str, Any]) -> float:
    text = (symbol.get("text") or "").lower()
    return pattern_line_score(pattern, text)


def pattern_line_score(pattern: str, text: str) -> float:
    text = text.lower()
    keywords = {
        "bounds_or_limit": ["limit", "max", "cap", "bound", "len", "percentile", "range"],
        "concurrency": ["lock", "unlock", "mutex", "atomic", "chan", "close", "wait", "done"],
        "error_handling": ["err", "error", "return", "panic", "recover"],
        "storage_cache": ["cache", "freezer", "database", "sync", "flush", "write", "read"],
        "test_change": ["test", "assert", "require", "expected", "want"],
        "generic": [],
    }.get(pattern, [])
    if not keywords:
        return 0.2
    hits = sum(1 for keyword in keywords if keyword in text)
    return min(1.0, hits / 3)


def is_boilerplate_statement(text: str) -> bool:
    stripped = text.strip()
    if any(stripped.startswith(prefix) for prefix in BOILERPLATE_PREFIXES):
        return True
    if stripped in {"{", "}", "},", ")", "nil", "true", "false"}:
        return True
    return False


class LLMClient:
    def __init__(self, timeout: int = 30) -> None:
        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("BASE_URL")
        model = os.environ.get("MODEL")
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.timeout = timeout
        self.last_error = ""
        self.enabled = bool(self.api_key and self.base_url and self.model)

    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.last_error = ""
        if not self.enabled:
            self.last_error = "disabled"
            return None
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a code propagation localization evaluator. "
                        "Return strict JSON only. Do not explain or show reasoning. "
                        "Rerank provided candidates; do not invent paths or lines."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
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
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self.last_error = f"HTTPError {exc.code}"
            return None
        except (urllib.error.URLError, TimeoutError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        except json.JSONDecodeError as exc:
            self.last_error = f"JSONDecodeError: {exc}"
            return None
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        parsed = extract_json_object(content)
        if parsed is None:
            self.last_error = f"unparseable content: {content[:200]}"
        return parsed


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


def normalize_statement_set(texts: list[str]) -> set[str]:
    out = set()
    for text in texts:
        norm = " ".join(TOKEN_RE.findall((text or "").lower()))
        if len(norm) >= 8:
            out.add(norm)
    return out


def line_hits_ranges(path: str, line: int, ranges: dict[str, list[tuple[int, int]]], tolerance: int) -> bool:
    return any(start - tolerance <= line <= end + tolerance for start, end in ranges.get(path, []))


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    file_hits = sum(1 for r in ok if r["metrics"]["file_topk_hit"])
    statement_hits = sum(1 for r in ok if r["metrics"]["statement_topk_hit"])
    hard_file = [r for r in ok if r.get("hardness", {}).get("file_worked_despite_path_change")]
    hard_stmt = [r for r in ok if r.get("hardness", {}).get("statement_worked_despite_text_change")]
    return {
        "evaluated_pairs": len(ok),
        "skipped_pairs": len(results) - len(ok),
        "file_topk_hits": file_hits,
        "file_topk_hit_rate": round(file_hits / len(ok), 4) if ok else None,
        "statement_topk_hits": statement_hits,
        "statement_topk_hit_rate": round(statement_hits / len(ok), 4) if ok else None,
        "file_hits_with_changed_paths": len(hard_file),
        "statement_hits_with_changed_text": len(hard_stmt),
    }


def hard_case_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for result in results:
        if result.get("status") != "ok":
            continue
        hardness = result.get("hardness", {})
        metrics = result.get("metrics", {})
        interesting = (
            hardness.get("path_not_exact")
            or hardness.get("basename_not_exact")
            or hardness.get("statement_text_not_exact")
            or not metrics.get("file_topk_hit")
            or not metrics.get("statement_topk_hit")
        )
        if not interesting:
            continue
        target = result.get("target", {})
        propagator = result.get("propagator", {})
        prediction = result.get("prediction", {})
        cases.append(
            {
                "index": result.get("index"),
                "source": {
                    "repo": propagator.get("repo"),
                    "kind": propagator.get("kind"),
                    "number": propagator.get("number"),
                    "title": propagator.get("title"),
                    "files": propagator.get("file_names"),
                },
                "target": {
                    "repo": target.get("repo"),
                    "kind": target.get("kind"),
                    "number": target.get("number"),
                    "title": target.get("title"),
                    "files": target.get("file_names"),
                },
                "metrics": metrics,
                "hardness": hardness,
                "top_files": prediction.get("target_files", [])[:5],
                "top_statements": prediction.get("target_statements", [])[:5],
                "llm_used": bool(prediction.get("llm_used")),
            }
        )
    return cases


def make_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary": summarize(results),
        "hard_cases": hard_case_summary(results),
        "results": results,
    }


def write_payload(output: Path, results: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(make_payload(results), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="sample_30_pairs.json")
    parser.add_argument("--output", default="outputs/qa_mcts_sample20.json")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--max-nodes", type=int, default=15)
    parser.add_argument("--top-files", type=int, default=8)
    parser.add_argument("--top-statements", type=int, default=10)
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI-compatible LLM reranker from .env/env vars.")
    parser.add_argument("--llm-hard-only", action="store_true", help="Call LLM only for ambiguous/harder candidates.")
    parser.add_argument("--llm-timeout", type=int, default=30, help="Timeout in seconds for one LLM reranking call.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write output after this many processed pairs.")
    parser.add_argument("--indices", default="", help="Comma-separated 1-based input indices to run, e.g. 4,16,18.")
    return parser.parse_args()


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def main() -> int:
    args = parse_args()
    load_dotenv()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    selected_indices = {
        int(part.strip())
        for part in args.indices.split(",")
        if part.strip().isdigit() and int(part.strip()) > 0
    }
    if selected_indices:
        pairs = [pair for index, pair in enumerate(pairs, start=1) if index in selected_indices]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    llm = LLMClient(timeout=args.llm_timeout) if args.use_llm else None
    if args.use_llm and (not llm or not llm.enabled):
        print(
            "LLM reranker requested but API_KEY, BASE_URL, or MODEL is not configured; "
            "continuing without LLM.",
            flush=True,
        )
        llm = None
    results = []
    output = Path(args.output)
    run_items = [
        (index, pair)
        for index, pair in enumerate(json.loads(Path(args.input).read_text(encoding="utf-8")), start=1)
        if not selected_indices or index in selected_indices
    ]
    if args.max_pairs:
        run_items = run_items[: args.max_pairs]
    for offset, (index, pair) in enumerate(run_items, start=1):
        runner = PropagationQAMCTS(
            client,
            max_nodes=args.max_nodes,
            top_files=args.top_files,
            top_statements=args.top_statements,
            llm=llm,
            llm_hard_only=args.llm_hard_only,
        )
        try:
            result = runner.run_pair(pair, index)
        except Exception as exc:
            result = {"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(f"[{offset}/{len(run_items)}; input_index={index}] {result['status']}: {result.get('reason', '')}", flush=True)
        if args.checkpoint_every and offset % args.checkpoint_every == 0:
            write_payload(output, results)
    payload = make_payload(results)
    write_payload(output, results)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
