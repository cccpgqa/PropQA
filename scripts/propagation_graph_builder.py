#!/usr/bin/env python3
"""Build graph-style evidence for propagation localization experiments.

This is the graph-oriented successor to the simple baseline in
`propagation_detector.py`. It builds a repository/file/symbol/statement graph
for the propagator change and the target repository before the target fix.

The graph is intentionally JSON-first so a QA agent, planner, or MCTS loop can
consume it as an external environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from propagation_detector import (
    GitHubClient,
    Side,
    iter_patch_statements,
    parsed_sides,
    path_score,
    remote_changed_files,
    resolve_side,
    token_similarity,
)


GO_DECL_RE = re.compile(
    r"^\s*(?P<kind>func|type|var|const)\s+"
    r"(?:(?P<receiver>\([^)]*\))\s*)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)


def make_node_id(*parts: object) -> str:
    return "::".join(str(part).replace("::", "__") for part in parts)


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []
        self._edge_seen: set[tuple[str, str, str]] = set()

    def add_node(self, node_id: str, kind: str, **attrs: Any) -> str:
        node = {"id": node_id, "kind": kind}
        node.update(attrs)
        self.nodes[node_id] = node
        return node_id

    def add_edge(self, source: str, target: str, kind: str) -> None:
        key = (source, target, kind)
        if key in self._edge_seen:
            return
        self._edge_seen.add(key)
        self.edges.append({"source": source, "target": target, "kind": kind})

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": self.edges}


def build_repo_structure_graph(repo: str, tree_paths: list[str], changed_paths: set[str]) -> Graph:
    graph = Graph()
    repo_id = graph.add_node(make_node_id("repo", repo), "repo", repo=repo)
    dirs: set[str] = set()
    for path in tree_paths:
        parts = path.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    for directory in sorted(dirs):
        dir_id = graph.add_node(make_node_id(repo, "dir", directory), "dir", path=directory)
        parent = repo_id if "/" not in directory else make_node_id(repo, "dir", directory.rsplit("/", 1)[0])
        graph.add_edge(parent, dir_id, "contains")
    for path in sorted(tree_paths):
        file_id = graph.add_node(
            make_node_id(repo, "file", path),
            "file",
            path=path,
            name=path.rsplit("/", 1)[-1],
            extension=file_extension(path),
            changed_in_patch=path in changed_paths,
        )
        parent = repo_id if "/" not in path else make_node_id(repo, "dir", path.rsplit("/", 1)[0])
        graph.add_edge(parent, file_id, "contains")
    return graph


def enrich_file_ast(graph: Graph, repo: str, path: str, content: str, max_statement_nodes: int) -> None:
    file_id = make_node_id(repo, "file", path)
    symbols = parse_go_symbols(content)
    lines = content.splitlines()
    for symbol in symbols:
        symbol_id = graph.add_node(
            make_node_id(repo, "symbol", path, symbol["name"], symbol["start_line"]),
            "symbol",
            path=path,
            name=symbol["name"],
            symbol_kind=symbol["kind"],
            receiver=symbol.get("receiver"),
            start_line=symbol["start_line"],
            end_line=symbol["end_line"],
            signature=symbol["signature"],
        )
        graph.add_edge(file_id, symbol_id, "defines")
        count = 0
        for line_no in range(symbol["start_line"], min(symbol["end_line"], len(lines)) + 1):
            text = lines[line_no - 1].strip()
            if not is_statement_like(text):
                continue
            stmt_id = graph.add_node(
                make_node_id(repo, "stmt", path, line_no),
                "statement",
                path=path,
                line=line_no,
                text=text[:300],
            )
            graph.add_edge(symbol_id, stmt_id, "contains_statement")
            count += 1
            if count >= max_statement_nodes:
                break


def parse_go_symbols(content: str) -> list[dict[str, Any]]:
    lines = content.splitlines()
    symbols: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        match = GO_DECL_RE.match(line)
        if not match:
            continue
        kind = match.group("kind")
        name = match.group("name")
        receiver = match.group("receiver")
        end_line = find_decl_end(lines, idx)
        symbols.append(
            {
                "kind": "method" if kind == "func" and receiver else kind,
                "name": name,
                "receiver": receiver,
                "start_line": idx,
                "end_line": end_line,
                "signature": line.strip()[:300],
            }
        )
    return symbols


def find_decl_end(lines: list[str], start_line: int) -> int:
    balance = 0
    seen_open = False
    for idx in range(start_line, min(len(lines), start_line + 400) + 1):
        line = strip_line_comment(lines[idx - 1])
        balance += line.count("{")
        if "{" in line:
            seen_open = True
        balance -= line.count("}")
        if seen_open and balance <= 0:
            return idx
        if not seen_open and idx > start_line and lines[idx - 1].strip() == "":
            return idx - 1
    return min(len(lines), start_line + 80)


def strip_line_comment(line: str) -> str:
    pos = line.find("//")
    return line if pos < 0 else line[:pos]


def is_statement_like(text: str) -> bool:
    if not text or text in {"{", "}", "},", ");"}:
        return False
    if text.startswith("//") or text.startswith("*"):
        return False
    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text)) >= 2


def file_extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def changed_paths(files: Iterable[dict[str, Any]]) -> list[str]:
    return [item["filename"] for item in files if item.get("filename")]


def build_pair_graph_package(
    client: GitHubClient,
    pair: dict[str, Any],
    index: int,
    top_files: int,
    max_ast_files: int,
    max_statement_nodes: int,
) -> dict[str, Any]:
    sides = parsed_sides(pair)
    if sides is None:
        return {"index": index, "status": "skipped", "reason": "missing side metadata"}
    source_side, infestor_side = (resolve_side(client, side) for side in sides)
    ordered = order_pair(source_side, infestor_side)
    if ordered is None:
        return {"index": index, "status": "skipped", "reason": "missing base commit or date metadata"}
    propagator, target = ordered

    propagator_files = remote_changed_files(client, propagator)
    target_files = remote_changed_files(client, target)
    propagator_paths = changed_paths(propagator_files)
    target_paths = changed_paths(target_files)
    target_tree = client.tree_paths(target.repo, target.base_sha)

    target_graph = build_repo_structure_graph(target.repo, target_tree, set(target_paths))
    source_graph = build_repo_structure_graph(propagator.repo, propagator_paths, set(propagator_paths))

    ranked_files = rank_target_files_for_graph(propagator_paths, target_tree, top_files)
    ast_paths = list(dict.fromkeys(propagator_paths[:max_ast_files] + [item["path"] for item in ranked_files[:max_ast_files]]))
    for path in ast_paths:
        if path in propagator_paths:
            content = client.file_at(propagator.repo, propagator.base_sha, path)
            if content:
                enrich_file_ast(source_graph, propagator.repo, path, content, max_statement_nodes)
        if path in target_tree:
            content = client.file_at(target.repo, target.base_sha, path)
            if content:
                enrich_file_ast(target_graph, target.repo, path, content, max_statement_nodes)

    patch_nodes = patch_statement_nodes(propagator.repo, propagator_files)
    qa_prompt = build_qa_prompt(propagator, target, propagator_paths, ranked_files, patch_nodes)
    return {
        "index": index,
        "status": "ok",
        "propagator": dataclasses.asdict(propagator),
        "target": dataclasses.asdict(target),
        "question": qa_prompt,
        "propagator_changed_files": propagator_paths,
        "target_ground_truth_files": target_paths,
        "candidate_target_files": ranked_files,
        "propagator_patch_statements": patch_nodes,
        "source_change_graph": source_graph.to_dict(),
        "target_repo_graph": target_graph.to_dict(),
        "agent_action_space": [
            "Finish",
            "FindFile",
            "FindSymbol",
            "FindStatement",
            "ViewCode",
            "SemanticSearch",
            "ExpandNeighbors",
        ],
    }


def order_pair(source: Side, infestor: Side) -> tuple[Side, Side] | None:
    if not (source.base_sha and source.merged_at and infestor.base_sha and infestor.merged_at):
        return None
    return (source, infestor) if source.merged_at <= infestor.merged_at else (infestor, source)


def rank_target_files_for_graph(source_paths: list[str], target_tree: list[str], top_k: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_path in source_paths:
        best = sorted(
            (
                {
                    "path": target_path,
                    "score": round(path_score(source_path, target_path), 6),
                    "matched_source_path": source_path,
                }
                for target_path in target_tree
            ),
            key=lambda item: (-item["score"], item["path"]),
        )[:top_k]
        candidates.extend(best)
    candidates.sort(key=lambda item: (-item["score"], item["path"], item["matched_source_path"]))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in candidates:
        if item["path"] in seen:
            continue
        seen.add(item["path"])
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def patch_statement_nodes(repo: str, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for stmt in iter_patch_statements(files, include_context=True):
        nodes.append(
            {
                "id": make_node_id(repo, "patch_stmt", stmt.path, stmt.line_no, stmt.kind),
                "kind": stmt.kind,
                "path": stmt.path,
                "line": stmt.line_no,
                "text": stmt.text[:300],
            }
        )
    return nodes


def build_qa_prompt(
    propagator: Side,
    target: Side,
    source_paths: list[str],
    ranked_files: list[dict[str, Any]],
    patch_nodes: list[dict[str, Any]],
) -> str:
    source_preview = ", ".join(source_paths[:8])
    target_preview = ", ".join(item["path"] for item in ranked_files[:8])
    stmt_preview = " | ".join(item["text"] for item in patch_nodes[:5])
    return (
        f"Given a propagated change from {propagator.repo} to {target.repo}, locate the likely target "
        f"file(s), symbol(s), and statement(s) in the target repository before the fix. "
        f"Source changed files: {source_preview}. "
        f"Initial target file candidates from structure/name matching: {target_preview}. "
        f"Important source patch statements: {stmt_preview}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="sample_30_pairs.json")
    parser.add_argument("--output", default="outputs/graph_package_pair1.json")
    parser.add_argument("--pair-index", type=int, default=1, help="1-based pair index.")
    parser.add_argument("--top-files", type=int, default=12)
    parser.add_argument("--max-ast-files", type=int, default=8)
    parser.add_argument("--max-statement-nodes", type=int, default=80)
    parser.add_argument("--cache-dir", default=".cache/github")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pair = pairs[args.pair_index - 1]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    package = build_pair_graph_package(
        client,
        pair,
        index=args.pair_index,
        top_files=args.top_files,
        max_ast_files=args.max_ast_files,
        max_statement_nodes=args.max_statement_nodes,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": package["status"],
                "source_graph_nodes": len(package.get("source_change_graph", {}).get("nodes", [])),
                "target_graph_nodes": len(package.get("target_repo_graph", {}).get("nodes", [])),
                "candidate_target_files": len(package.get("candidate_target_files", [])),
                "patch_statements": len(package.get("propagator_patch_statements", [])),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
