#!/usr/bin/env python3
"""Generate propagation graph packages using Prometheus FileGraphBuilder.

This adapter reuses `Prometheus-main/prometheus/graph/file_graph_builder.py`
for file-level AST/text graph construction. It downloads only the files needed
for a pair, writes them to an output workspace, invokes Prometheus on those
local files, and serializes graph nodes/edges to JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from propagation_detector import GitHubClient, path_score


PROMETHEUS_ROOT = Path("Prometheus-main")


def import_prometheus_graph_builder():
    sys.path.insert(0, str(PROMETHEUS_ROOT.resolve()))
    try:
        import tree_sitter_language_pack

        ts_cache = Path(".cache/tree-sitter-language-pack").resolve()
        ts_cache.mkdir(parents=True, exist_ok=True)
        tree_sitter_language_pack.configure(cache_dir=str(ts_cache))

        from prometheus.graph.file_graph_builder import FileGraphBuilder
        from prometheus.graph.graph_types import (
            FileNode,
            KnowledgeGraphEdge,
            KnowledgeGraphNode,
        )
    except SyntaxError as exc:
        raise SystemExit(
            "Prometheus requires Python 3.10+ because it uses match/case syntax. "
            "Current Python cannot import Prometheus-main. Install/use Python 3.10+ "
            "and the dependencies from Prometheus-main/pyproject.toml, then rerun this script."
        ) from exc
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Prometheus dependencies are not installed. Install the dependencies from "
            "Prometheus-main/pyproject.toml, then rerun this script. Missing: "
            f"{exc.name}"
        ) from exc
    return FileGraphBuilder, FileNode, KnowledgeGraphNode


def side_repo(side: dict[str, Any]) -> str:
    project = side.get("project") or ""
    mapping = {
        "ethereum_go-ethereum": "ethereum/go-ethereum",
        "bnb-chain_bsc": "bnb-chain/bsc",
        "0xPolygon_bor": "0xPolygon/bor",
        "ava-labs_avalanchego": "ava-labs/avalanchego",
        "MystenLabs_sui": "MystenLabs/sui",
        "aptos-labs_aptos-core": "aptos-labs/aptos-core",
    }
    if project in mapping:
        return mapping[project]
    if "_" in project:
        owner, name = project.split("_", 1)
        return f"{owner}/{name}"
    return project


def side_base_sha(side: dict[str, Any]) -> str:
    resolved = side.get("resolved_by")
    if isinstance(resolved, dict) and resolved.get("base_commit_sha"):
        return resolved["base_commit_sha"]
    return side.get("base_commit_sha") or ""


def side_files(side: dict[str, Any]) -> list[str]:
    resolved = side.get("resolved_by")
    if isinstance(resolved, dict) and resolved.get("file_names"):
        return list(resolved.get("file_names") or [])
    return list(side.get("file_names") or [])


def rank_target_candidates(source_files: list[str], target_files: list[str], top_k: int) -> list[str]:
    if not source_files:
        return target_files[:top_k]
    scored = []
    for target in target_files:
        best = max(path_score(source, target) for source in source_files)
        scored.append((best, target))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [target for _, target in scored[:top_k]]


def write_remote_file(client: GitHubClient, repo: str, sha: str, path: str, root: Path) -> Path | None:
    text = client.file_at(repo, sha, path)
    if text is None:
        return None
    local = root / path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text, encoding="utf-8", errors="replace")
    return local


def graph_file(builder: Any, FileNode: Any, KnowledgeGraphNode: Any, file_path: Path, relative_path: str, next_id: int) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    parent = KnowledgeGraphNode(next_id, FileNode(basename=file_path.name, relative_path=relative_path))
    next_id += 1
    nodes = [serialize_node(parent)]
    edges: list[dict[str, Any]] = []
    if builder.supports_file(file_path):
        next_id, kg_nodes, kg_edges = builder.build_file_graph(parent, file_path, next_id)
        nodes.extend(serialize_node(node) for node in kg_nodes)
        edges.extend(serialize_edge(edge) for edge in kg_edges)
    return next_id, nodes, edges


def serialize_node(node: Any) -> dict[str, Any]:
    payload = dataclasses.asdict(node.node) if dataclasses.is_dataclass(node.node) else {}
    payload["node_id"] = node.node_id
    payload["node_kind"] = type(node.node).__name__
    return payload


def serialize_edge(edge: Any) -> dict[str, Any]:
    return {
        "source": edge.source.node_id,
        "target": edge.target.node_id,
        "type": str(edge.type),
    }


def build_pair_package(
    client: GitHubClient,
    pair: dict[str, Any],
    pair_index: int,
    output_workspace: Path,
    top_target_files: int,
    max_ast_depth: int,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    FileGraphBuilder, FileNode, KnowledgeGraphNode = import_prometheus_graph_builder()
    builder = FileGraphBuilder(max_ast_depth, chunk_size, chunk_overlap)
    source = pair["Source"]
    target = pair["Infestor"]
    source_files = side_files(source)
    target_files = side_files(target)
    selected_target_files = rank_target_candidates(source_files, target_files, top_target_files)

    workspace = output_workspace / f"pair_{pair_index:04d}"
    source_root = workspace / "source_before"
    target_root = workspace / "target_before"
    source_repo = side_repo(source)
    target_repo = side_repo(target)
    source_sha = side_base_sha(source)
    target_sha = side_base_sha(target)

    next_id = 0
    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    materialized: list[dict[str, str]] = []
    for role, repo, sha, root, paths in [
        ("source_before", source_repo, source_sha, source_root, source_files),
        ("target_before", target_repo, target_sha, target_root, selected_target_files),
    ]:
        if not sha:
            continue
        for path in paths:
            local = write_remote_file(client, repo, sha, path, root)
            if local is None:
                continue
            rel = f"{role}/{path}"
            next_id, nodes, edges = graph_file(builder, FileNode, KnowledgeGraphNode, local, rel, next_id)
            graph_nodes.extend(nodes)
            graph_edges.extend(edges)
            materialized.append({"role": role, "repo": repo, "sha": sha, "path": path, "local_path": str(local)})

    return {
        "pair_index": pair_index,
        "source": source,
        "target": target,
        "selected_target_files": selected_target_files,
        "materialized_files": materialized,
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "builder": {
            "name": "Prometheus FileGraphBuilder",
            "max_ast_depth": max_ast_depth,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/URL_Results_detection_subset.json")
    parser.add_argument("--pair-index", type=int, default=1)
    parser.add_argument("--output", default="outputs/prometheus_graph_pair1.json")
    parser.add_argument("--workspace", default="outputs/prometheus_graph_workspace")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--top-target-files", type=int, default=10)
    parser.add_argument("--max-ast-depth", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pair = pairs[args.pair_index - 1]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    package = build_pair_package(
        client,
        pair,
        args.pair_index,
        Path(args.workspace),
        args.top_target_files,
        args.max_ast_depth,
        args.chunk_size,
        args.chunk_overlap,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "nodes": len(package["graph"]["nodes"]),
                "edges": len(package["graph"]["edges"]),
                "materialized_files": len(package["materialized_files"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
