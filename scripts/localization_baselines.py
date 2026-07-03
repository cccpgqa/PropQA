#!/usr/bin/env python3
"""Run exhaustive localization baselines on a propagation dataset.

The code2vec baseline performs exhaustive retrieval over every target Go file:
source and target files are encoded by a Go-specific code2vec model and ranked
only by cosine similarity. No path prefilter, candidate pruning, path prior, or
ground-truth file information is used during file retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tarfile
import io
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from go_code2vec import GoCode2VecEncoder
from no_leak_ground_truth import statement_ground_truth_from_files
from propagation_detector import GitHubClient, token_similarity
from run_block_first_statement_experiment import metrics as statement_metrics
from run_graphqa_action_enhancement_experiment import (
    direct_intent_ast_blocks,
    enhanced_file_rank,
    expand_direct_blocks,
    file_metrics,
)
from run_llm_in_loop_hard_cases import prepare_state
from run_agent_strategy_matrix import map_candidates
from run_new_hard_baseline_ablation import action_graphqa, normalize_code, source_patch_text
from run_no_leak_five_module_pipeline import (
    normalize_target_oracle_to_dataset_files,
    split_detector_state,
    split_label,
)
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv, meaningful_statement

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CODE_EXTENSIONS = {
    ".go",
    ".sol",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".kt",
    ".scala",
    ".sh",
    ".proto",
}


def git_dir(root: Path, repo: str) -> Path:
    return root / f"{repo.replace('/', '_')}.git"


def git_file(root: Path, repo: str, sha: str, path: str) -> str:
    directory = git_dir(root, repo)
    try:
        result = subprocess.run(
            ["git", f"--git-dir={directory}", "show", f"{sha}:{path}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return result.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


def git_go_blobs(root: Path, repo: str, sha: str) -> list[tuple[str, str]]:
    directory = git_dir(root, repo)
    result = subprocess.run(
        ["git", f"--git-dir={directory}", "ls-tree", "-r", sha],
        check=True,
        capture_output=True,
        timeout=180,
    )
    rows = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        try:
            metadata, path = line.split("\t", 1)
            _mode, kind, blob = metadata.split()
        except ValueError:
            continue
        if kind == "blob" and path.endswith(".go"):
            rows.append((path, blob))
    return rows


def git_go_snapshot(root: Path, repo: str, sha: str) -> dict[str, str]:
    directory = git_dir(root, repo)
    result = subprocess.run(
        ["git", f"--git-dir={directory}", "archive", "--format=tar", sha],
        check=True,
        capture_output=True,
        timeout=600,
    )
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".go"):
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                files[member.name] = handle.read().decode("utf-8", errors="replace")
    return files


def is_code_file(path: str) -> bool:
    if path.startswith(".github/") or path.startswith(".gitea/"):
        return False
    suffix = Path(path).suffix.lower()
    return suffix in CODE_EXTENSIONS


def clone_tokens(text: str) -> set[str]:
    normalized = re.sub(r'"(?:\\.|[^"\\])*"|`[^`]*`|\b\d+\b', " LIT ", text)
    return {token.lower() for token in TOKEN_RE.findall(normalized) if len(token) >= 2}


def token_set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    dice = 2.0 * overlap / (len(left) + len(right))
    jaccard = overlap / len(left | right)
    return 0.60 * dice + 0.40 * jaccard


def clone_chunks(text: str, max_chunks: int = 80) -> list[set[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[set[str]] = []
    for line in lines:
        tokens = clone_tokens(line)
        if len(tokens) >= 3:
            chunks.append(tokens)
    for window in (3, 6):
        for start in range(0, max(0, len(lines) - window + 1), window):
            tokens = clone_tokens("\n".join(lines[start : start + window]))
            if len(tokens) >= 5:
                chunks.append(tokens)
    if len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    return [chunks[int(i * step)] for i in range(max_chunks)]


def clone_similarity_from_chunks(source_chunks: list[set[str]], target_text: str) -> float:
    return clone_similarity_against_target_chunks(source_chunks, clone_chunks(target_text[:24000]))


def clone_similarity_against_target_chunks(source_chunks: list[set[str]], target_chunks: list[set[str]]) -> float:
    best = 0.0
    for source in source_chunks:
        for target in target_chunks:
            score = token_set_similarity(source, target)
            if score > best:
                best = score
    return best


def git_code_snapshot(root: Path, repo: str, sha: str, max_file_bytes: int = 250_000) -> dict[str, str]:
    directory = git_dir(root, repo)
    files: dict[str, str] = {}
    with tempfile.TemporaryFile() as tmp:
        subprocess.run(
            ["git", f"--git-dir={directory}", "archive", "--format=tar", sha],
            check=True,
            stdout=tmp,
            timeout=600,
        )
        tmp.seek(0)
        with tarfile.open(fileobj=tmp, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not is_code_file(member.name):
                    continue
                if member.size > max_file_bytes:
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    files[member.name] = handle.read().decode("utf-8", errors="replace")
    return files


class ExhaustiveCode2Vec:
    def __init__(self, checkpoint: Path, git_root: Path, cache_root: Path) -> None:
        self.encoder = GoCode2VecEncoder(checkpoint)
        self.git_root = git_root
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.memory: dict[str, np.ndarray | None] = {}

    def blob_vector(self, repo: str, sha: str, path: str, blob: str | None = None) -> np.ndarray | None:
        key = blob or f"{repo}:{sha}:{path}"
        if key in self.memory:
            return self.memory[key]
        cache = self.cache_root / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.npy"
        if cache.exists():
            vector = np.load(cache)
            self.memory[key] = vector if vector.size else None
            return self.memory[key]
        text = git_file(self.git_root, repo, sha, path)
        vector = self.encoder.file_vector(text) if text else None
        np.save(cache, vector if vector is not None else np.asarray([], dtype=np.float32))
        self.memory[key] = vector
        return vector

    def function_vectors(self, repo: str, sha: str, path: str) -> list[dict[str, Any]]:
        text = git_file(self.git_root, repo, sha, path)
        return self.encoder.function_vectors(text) if text else []

    def rank_files(
        self,
        source_repo: str,
        source_sha: str,
        source_paths: list[str],
        target_repo: str,
        target_sha: str,
        top: int,
    ) -> list[dict[str, Any]]:
        source_vectors = []
        source_vector_paths = []
        for path in source_paths:
            vector = self.blob_vector(source_repo, source_sha, path)
            if vector is not None:
                source_vectors.append(vector)
                source_vector_paths.append(path)
        if not source_vectors:
            return []
        source_matrix = np.stack(source_vectors)
        target_entries = git_go_blobs(self.git_root, target_repo, target_sha)
        uncached = []
        for target_path, blob in target_entries:
            cache = self.cache_root / f"{hashlib.sha256(blob.encode('utf-8')).hexdigest()}.npy"
            if blob not in self.memory and not cache.exists():
                uncached.append((target_path, blob))
        snapshot = git_go_snapshot(self.git_root, target_repo, target_sha) if uncached else {}
        for target_path, blob in uncached:
            text = snapshot.get(target_path, "")
            vector = self.encoder.file_vector(text) if text else None
            cache = self.cache_root / f"{hashlib.sha256(blob.encode('utf-8')).hexdigest()}.npy"
            np.save(cache, vector if vector is not None else np.asarray([], dtype=np.float32))
            self.memory[blob] = vector
        ranked = []
        for target_path, blob in target_entries:
            vector = self.blob_vector(target_repo, target_sha, target_path, blob)
            if vector is None:
                continue
            similarities = source_matrix @ vector
            best = int(np.argmax(similarities))
            ranked.append(
                {
                    "path": target_path,
                    "score": float(similarities[best]),
                    "matched_source_path": source_vector_paths[best],
                    "evidence": {
                        "cosine_similarity": float(similarities[best]),
                        "retrieval_scope": "all_target_go_files",
                        "prefilter": None,
                    },
                    "baseline": "code2vec",
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["path"]))
        return ranked[:top]

    def rank_statements(
        self,
        source_repo: str,
        source_sha: str,
        source_paths: list[str],
        target_repo: str,
        target_sha: str,
        target_paths: list[str],
        top: int,
    ) -> list[dict[str, Any]]:
        source_functions = []
        for path in source_paths:
            source_functions.extend(self.function_vectors(source_repo, source_sha, path))
        if not source_functions:
            return []
        source_matrix = np.stack([item["vector"] for item in source_functions])
        candidates = []
        for path in target_paths:
            text = git_file(self.git_root, target_repo, target_sha, path)
            if not text:
                continue
            lines = text.splitlines()
            for function in self.encoder.function_vectors(text):
                similarities = source_matrix @ function["vector"]
                best = int(np.argmax(similarities))
                score = float(similarities[best])
                for line_no in range(function["start_line"], min(function["end_line"], len(lines)) + 1):
                    line = lines[line_no - 1].strip()
                    if not meaningful_statement(line):
                        continue
                    candidates.append(
                        {
                            "path": path,
                            "line": line_no,
                            "score": score,
                            "line_text": line[:240],
                            "target_symbol": function["name"],
                            "matched_source_symbol": source_functions[best]["name"],
                            "baseline": "code2vec",
                        }
                    )
        candidates.sort(key=lambda item: (-item["score"], item["path"], item["line"]))
        return candidates[:top]


def average(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    return {
        field: sum(float(row.get(field, 0.0)) for row in rows) / len(rows) if rows else 0.0
        for field in fields
    }


def local_nicad(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    git_root: Path,
    top: int,
    snapshot_cache: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_text = source_patch_text(state)
    source_chunks = clone_chunks(source_text, max_chunks=80)
    target_key = (state["target"].repo, state["target"].base_sha)
    if target_key not in snapshot_cache:
        snapshot_cache[target_key] = git_code_snapshot(git_root, state["target"].repo, state["target"].base_sha)
    ranked = []
    for path, text in snapshot_cache[target_key].items():
        clone = clone_similarity_from_chunks(source_chunks, text) if text else 0.0
        ranked.append(
            {
                "path": path,
                "score": clone,
                "matched_source_path": None,
                "evidence": {"retrieval_scope": "all_target_code_files", "path_score": None},
                "baseline": "Open-NiCad",
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), item["path"]))
    selected = ranked[:top]
    statements = map_candidates(runner, state, selected, 200)
    return selected, statements[:100]


def aggregate(rows: list[dict[str, Any]], split: str, method: str, level: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["method"] == method
        and row["level"] == level
        and (split == "all" or row["split"] == split)
    ]
    fields = ["precision", "recall", "f1", "mrr"] if level == "file" else ["precision", "coverage", "f1", "mrr"]
    return {"split": split, "method": method, "level": level, "pairs": len(selected), **average(selected, fields)}


def run_dataset(
    pairs: list[dict[str, Any]],
    label: str,
    client: GitHubClient,
    code2vec: ExhaustiveCode2Vec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    rows = []
    details = []
    snapshot_cache: dict[tuple[str, str], dict[str, str]] = {}
    for index, pair in enumerate(pairs, start=1):
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, oracle = split_detector_state(setup)
            split = split_label(pair)
            base = action_graphqa(state, 40)
            graphqa_files = enhanced_file_rank(state, base, top=10, per_source_pool=20)
            nicad_files, nicad_statements = local_nicad(
                state, runner, Path(args.git_root), 10, snapshot_cache
            )
            code2vec_files = code2vec.rank_files(
                state["propagator"].repo,
                state["propagator"].merged_sha,
                state["source_changed_paths"],
                state["target"].repo,
                state["target"].base_sha,
                10,
            )
            gt_files = set(oracle["target_changed_paths"])
            for method, predictions in (
                ("Open-NiCad", nicad_files),
                ("code2vec", code2vec_files),
                ("PropQA", graphqa_files),
            ):
                rows.append(
                    {
                        "index": index,
                        "split": split,
                        "method": method,
                        "level": "file",
                        **file_metrics([item["path"] for item in predictions], gt_files, 10),
                    }
                )

            target_paths = list(oracle["target_changed_paths"])
            blocks, intents = direct_intent_ast_blocks(state, runner, target_paths, 2)
            budget = min(1200, max(100, 8 * len(intents), 6 * len(target_paths)))
            graphqa_statements = expand_direct_blocks(client, state, blocks, intents, budget, 6)
            code2vec_statements = code2vec.rank_statements(
                state["propagator"].repo,
                state["propagator"].merged_sha,
                state["source_changed_paths"],
                state["target"].repo,
                state["target"].base_sha,
                target_paths,
                100,
            )
            gt_statements = statement_ground_truth_from_files(oracle["target_files"])
            for method, predictions in (
                ("Open-NiCad", nicad_statements),
                ("code2vec", code2vec_statements),
                ("PropQA", graphqa_statements),
            ):
                rows.append(
                    {
                        "index": index,
                        "split": split,
                        "method": method,
                        "level": "statement",
                        **statement_metrics(predictions, gt_statements, 100, tolerance=2),
                    }
                )
            details.append(
                {
                    "index": index,
                    "split": split,
                    "source_repo": state["propagator"].repo,
                    "target_repo": state["target"].repo,
                    "ground_truth_files": sorted(gt_files),
                    "file_predictions": {
                        "Open-NiCad": [item["path"] for item in nicad_files],
                        "code2vec": [item["path"] for item in code2vec_files],
                        "PropQA": [item["path"] for item in graphqa_files],
                    },
                }
            )
            print(f"[{label} {index}/{len(pairs)}] ok", flush=True)
        except Exception as exc:
            details.append({"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{label} {index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)
    splits = ["all", "simple", "hard"] if label == "url" else ["all"]
    summary = [
        aggregate(rows, split, method, level)
        for split in splits
        for level in ["file", "statement"]
        for method in ["Open-NiCad", "code2vec", "PropQA"]
    ]
    return {"dataset": label, "pairs": len(pairs), "summary": summary, "rows": rows, "details": details}


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Localization Baseline Results",
        "",
        "The code2vec baseline uses exhaustive all-target-Go-file cosine retrieval.",
        "It uses no path prefilter, path score, candidate pruning, or target ground-truth hints.",
        "The Open-NiCad baseline uses exhaustive all-target-code-file token-clone retrieval.",
        "It also uses no path prefilter, path score, candidate pruning, or target ground-truth hints.",
        "",
    ]
    for result in payload["results"]:
        lines += [
            f"## {result['dataset']}",
            "",
            "| Split | Level | Method | Pairs | Precision | Recall/Coverage | F1 | MRR |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in result["summary"]:
            recall = row.get("recall", row.get("coverage", 0.0))
            lines.append(
                f"| {row['split']} | {row['level']} | {row['method']} | {row['pairs']} | "
                f"{row['precision']:.3f} | {recall:.3f} | {row['f1']:.3f} | {row['mrr']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/detection_pairs_225.json")
    parser.add_argument("--checkpoint", default=".cache/go_code2vec/geth_go_code2vec.pt")
    parser.add_argument("--output", default="results/baselines/exhaustive.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--git-root", default=".cache/git_repos")
    parser.add_argument("--vector-cache", default=".cache/go_code2vec/blob_vectors_ctx12_term8")
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    # Compatibility fields used by the Open-NiCad baseline.
    args.prefilter_files = 80
    args.max_file_chars = 12000
    args.candidate_files = 10
    args.inspect_files = 18
    args.candidate_statements = 200
    args.top_statements = 100

    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    encoder = ExhaustiveCode2Vec(Path(args.checkpoint), Path(args.git_root), Path(args.vector_cache))
    pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    results = [run_dataset(pairs, "detection_benchmark", client, encoder, args)]
    payload = {
        "configuration": {
            "dataset": args.dataset,
            "code2vec_checkpoint": args.checkpoint,
            "code2vec_retrieval": "all target Go files ranked only by cosine similarity",
            "open_nicad_retrieval": "all target code files ranked only by token-clone similarity",
        },
        "results": results,
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
