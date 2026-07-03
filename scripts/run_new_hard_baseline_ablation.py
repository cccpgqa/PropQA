#!/usr/bin/env python3
"""Run baseline/action ablations on manually curated new hard samples.

The experiment uses runnable proxies for the requested baselines:
- dcard-clone: marked N/A because the repo is a web-app clone, not a detector.
- Open-NiCad: normalized near-miss clone retrieval.
- code2vec: AST/symbol/path-context retrieval.
- CodeBERT: embedding retrieval; uses local sentence-transformer fallback unless
  a CodeBERT-compatible model path is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_detection_tables import collect_rows, markdown_table
from propagation_detector import GitHubClient, jaccard, path_score, path_tokens, token_similarity
from run_agent_strategy_matrix import map_candidates, neighbor_expand, rank_files_from_evidence, round_robin_statements
from run_llm_in_loop_hard_cases import evaluate_prediction, prepare_state
from run_qa_mcts_small_sample import PropagationQAMCTS, dedupe_statement_predictions, load_dotenv

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
DEFAULT_MODEL = "D:/Python/Blockchain/sentenceTransformer/all-MiniLM-L12-v2"


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "") if len(t) >= 2}


def normalize_code(text: str) -> str:
    text = re.sub(r"//.*|/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r'"(?:\\.|[^"\\])*"|`[^`]*`|\b\d+\b', " LIT ", text)
    text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", " ID ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def source_patch_text(state: dict[str, Any]) -> str:
    return "\n".join(stmt.text for stmt in state.get("source_statements", [])[:120])


def safe_file_at(runner: PropagationQAMCTS, repo: str, sha: str, path: str) -> str:
    try:
        return runner.client.file_at(repo, sha, path) or ""
    except Exception:
        return ""


def local_git_dir(args: argparse.Namespace, repo: str) -> Path:
    return Path(getattr(args, "local_git_repo_cache", ".cache/git_repos")) / f"{repo.replace('/', '_')}.git"


def local_git_file(args: argparse.Namespace, repo: str, sha: str, path: str) -> str:
    git_dir = local_git_dir(args, repo)
    if not git_dir.exists():
        return ""
    try:
        result = subprocess.run(
            ["git", f"--git-dir={git_dir}", "show", f"{sha}:{path}"],
            check=True,
            capture_output=True,
            timeout=180,
        )
        return result.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


def local_git_snapshot_documents(
    args: argparse.Namespace,
    repo: str,
    sha: str,
) -> tuple[list[str], list[str]]:
    """Read a pre-fix repository snapshot from a local partial bare clone."""
    git_dir = local_git_dir(args, repo)
    if not git_dir.exists():
        return [], []
    try:
        result = subprocess.run(
            ["git", f"--git-dir={git_dir}", "archive", "--format=tar", sha],
            check=True,
            capture_output=True,
            timeout=getattr(args, "local_git_archive_timeout", 1800),
        )
        paths, docs = [], []
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not is_code_file(member.name):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                text = extracted.read().decode("utf-8", errors="replace")
                paths.append(member.name)
                docs.append(member.name + "\n" + text[: args.max_file_chars])
        return paths, docs
    except Exception:
        return [], []


def source_terms(state: dict[str, Any]) -> set[str]:
    out = set()
    for path in state.get("source_changed_paths", []):
        out |= path_tokens(path)
    out |= tokens(source_patch_text(state))
    title = state["propagator"].title + "\n" + state["propagator"].body + "\n" + state["propagator"].message
    out |= tokens(title)
    return {t for t in out if len(t) >= 3}


def path_prefilter(state: dict[str, Any], limit: int = 120) -> list[dict[str, Any]]:
    terms = source_terms(state)
    scored = {}
    for sp in state["source_changed_paths"]:
        for tp in state["target_tree"]:
            score = 0.68 * path_score(sp, tp) + 0.32 * jaccard(path_tokens(tp), terms)
            old = scored.get(tp)
            if old is None or score > old["score"]:
                scored[tp] = {"path": tp, "score": score, "matched_source_path": sp}
    return sorted(scored.values(), key=lambda x: (-x["score"], x["path"]))[:limit]


def action_graphqa(state: dict[str, Any], top: int) -> list[dict[str, Any]]:
    return path_prefilter(state, top)


def action_test_impact(state: dict[str, Any], candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_path = {c["path"]: dict(c) for c in candidates}
    source_has_tests = any("_test." in p or "/test" in p.lower() or "test/" in p.lower() for p in state["source_changed_paths"])
    changed_bases = {p.rsplit("/", 1)[-1].split(".")[0].replace("_test", "") for p in state["source_changed_paths"]}
    for path in state["target_tree"]:
        lower = path.lower()
        base = path.rsplit("/", 1)[-1].split(".")[0].replace("_test", "")
        if (source_has_tests or base in changed_bases) and ("_test." in lower or "/test" in lower or "test/" in lower):
            score = 0.58 + 0.22 * (base in changed_bases) + 0.20 * jaccard(path_tokens(path), source_terms(state))
            old = by_path.get(path)
            if old is None or score > old.get("score", 0):
                by_path[path] = {"path": path, "score": round(score, 6), "matched_source_path": next(iter(state["source_changed_paths"]), None), "action": "AskTestImpact"}
    return sorted(by_path.values(), key=lambda x: (-float(x["score"]), x["path"]))[:limit]


def changed_symbol_terms(state: dict[str, Any]) -> set[str]:
    text = source_patch_text(state)
    names = set()
    for pat in [r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)", r"\b(type|var|const)\s+([A-Za-z_][A-Za-z0-9_]*)", r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\("]:
        for match in re.finditer(pat, text):
            names.add(match.group(match.lastindex).lower())
    names |= {t for t in tokens(text) if len(t) >= 6}
    return names


def action_symbol_impact(state: dict[str, Any], candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    symbols = changed_symbol_terms(state)
    by_path = {c["path"]: dict(c) for c in candidates}
    for path in state["target_tree"]:
        score = jaccard(path_tokens(path), symbols | source_terms(state))
        if score <= 0.04:
            continue
        old = by_path.get(path)
        candidate = {"path": path, "score": round(0.46 + score, 6), "matched_source_path": next(iter(state["source_changed_paths"]), None), "action": "AskSymbolImpact"}
        if old is None or candidate["score"] > old.get("score", 0):
            by_path[path] = candidate
    return sorted(by_path.values(), key=lambda x: (-float(x["score"]), x["path"]))[:limit]


def action_deletion_interface_build_impact(state: dict[str, Any], candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    patch = source_patch_text(state).lower()
    deletion_signal = sum(1 for stmt in state.get("source_statements", []) if stmt.kind == "-")
    keywords = {"protocol", "handler", "flag", "config", "api", "rpc", "json", "marshal", "import", "package", "version", "gas", "lock", "error"}
    active = deletion_signal > 2 or any(k in patch for k in keywords)
    if not active:
        return candidates[:limit]
    by_path = {c["path"]: dict(c) for c in candidates}
    terms = source_terms(state) | keywords
    for path in state["target_tree"]:
        pterms = path_tokens(path)
        score = jaccard(pterms, terms)
        if score <= 0.035:
            continue
        bonus = 0.08 if any(k in path.lower() for k in ["protocol", "handler", "api", "rpc", "config", "flags", "json", "types"]) else 0.0
        candidate = {"path": path, "score": round(0.42 + score + bonus, 6), "matched_source_path": next(iter(state["source_changed_paths"]), None), "action": "AskDeletionInterfaceBuildImpact"}
        old = by_path.get(path)
        if old is None or candidate["score"] > old.get("score", 0):
            by_path[path] = candidate
    return sorted(by_path.values(), key=lambda x: (-float(x["score"]), x["path"]))[:limit]


def baseline_open_nicad(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    src_norm = normalize_code(source_patch_text(state))
    candidates = path_prefilter(state, args.prefilter_files)
    ranked = []
    for item in candidates:
        text = safe_file_at(runner, state["target"].repo, state["target"].base_sha, item["path"])
        score = 0.72 * token_similarity(src_norm, normalize_code(text[: args.max_file_chars])) + 0.28 * item["score"]
        ranked.append({**item, "score": round(score, 6), "baseline": "Open-NiCad-proxy"})
    ranked = sorted(ranked, key=lambda x: (-x["score"], x["path"]))[: args.candidate_files]
    statements = map_candidates(runner, state, ranked[: args.inspect_files], args.candidate_statements)
    return ranked, statements[: args.top_statements]


def baseline_code2vec(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbols = changed_symbol_terms(state)
    ranked = []
    for item in path_prefilter(state, args.prefilter_files):
        pterms = path_tokens(item["path"])
        context_score = jaccard(symbols | source_terms(state), pterms)
        score = 0.50 * item["score"] + 0.50 * context_score
        ranked.append({**item, "score": round(score, 6), "baseline": "code2vec-proxy"})
    ranked = sorted(ranked, key=lambda x: (-x["score"], x["path"]))[: args.candidate_files]
    statements = map_candidates(runner, state, ranked[: args.inspect_files], args.candidate_statements)
    statements = sorted(statements, key=lambda x: (-(float(x.get("symbol_score", 0)) + float(x.get("score", 0))), x["path"], int(x["line"])))
    return ranked, dedupe_statement_predictions(statements)[: args.top_statements]


class Embedder:
    def __init__(self, model_path: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_path = model_path
        self.model = SentenceTransformer(model_path)

    def score(self, query: str, docs: list[str]) -> list[float]:
        emb = self.model.encode([query] + docs, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        sims = emb[1:] @ emb[0]
        return [float(x) for x in sims]

    def encode(self, docs: list[str], batch_size: int = 32):
        return self.model.encode(docs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
    ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
    ".py", ".rs", ".sol", ".scala", ".kt", ".kts",
    ".cs", ".rb", ".php", ".swift", ".sh", ".bash",
    ".proto", ".toml", ".yaml", ".yml", ".json",
}
CODE_FILENAMES = {"go.mod", "go.sum", "cargo.toml", "cargo.lock", "makefile", "dockerfile"}


def is_code_file(path: str) -> bool:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    return suffix in CODE_EXTENSIONS or name in CODE_FILENAMES


def embedding_cache_path(args: argparse.Namespace, repo: str, sha: str, paths: list[str]) -> Path:
    cache_dir = Path(getattr(args, "semantic_embedding_cache", ".cache/semantic_embeddings"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(args, "embedding_model", DEFAULT_MODEL)
    key = json.dumps(
        {
            "repo": repo,
            "sha": sha,
            "model": model,
            "max_chars": args.max_file_chars,
            "paths": paths,
        },
        sort_keys=True,
    )
    return cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.npz"


def encode_target_repository(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    args: argparse.Namespace,
    embedder: Embedder,
) -> tuple[list[str], Any]:
    import numpy as np

    paths = [path for path in state["target_tree"] if is_code_file(path)]
    max_files = int(getattr(args, "semantic_max_target_files", 0) or 0)
    if max_files:
        paths = paths[:max_files]
    cache_path = embedding_cache_path(args, state["target"].repo, state["target"].base_sha, paths)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return [str(path) for path in cached["paths"].tolist()], cached["embeddings"]
    kept_paths, docs = local_git_snapshot_documents(args, state["target"].repo, state["target"].base_sha)
    allowed = set(paths)
    if kept_paths:
        filtered = [(path, doc) for path, doc in zip(kept_paths, docs) if path in allowed]
        kept_paths = [path for path, _ in filtered]
        docs = [doc for _, doc in filtered]
    else:
        docs = []
        kept_paths = []
        for path in paths:
            text = safe_file_at(runner, state["target"].repo, state["target"].base_sha, path)
            if not text:
                continue
            kept_paths.append(path)
            docs.append(path + "\n" + text[: args.max_file_chars])
    embeddings = embedder.encode(docs, batch_size=getattr(args, "semantic_batch_size", 32))
    np.savez_compressed(cache_path, paths=np.asarray(kept_paths), embeddings=embeddings)
    return kept_paths, embeddings


def baseline_semantic_embedding(
    state: dict[str, Any],
    runner: PropagationQAMCTS,
    args: argparse.Namespace,
    embedder: Embedder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Independent file embedding retrieval without path/PropQA prefiltering."""
    import numpy as np

    source_docs = []
    source_paths = []
    for path in state["source_changed_paths"]:
        text = local_git_file(args, state["propagator"].repo, state["propagator"].merged_sha, path)
        if not text:
            text = safe_file_at(runner, state["propagator"].repo, state["propagator"].merged_sha, path)
        if not text:
            patch_lines = [stmt.text for stmt in state.get("source_statements", []) if stmt.path == path]
            text = "\n".join(patch_lines)
        if not text:
            continue
        source_paths.append(path)
        source_docs.append(path + "\n" + text[: args.max_file_chars])
    if not source_docs:
        return [], []
    target_paths, target_embeddings = encode_target_repository(state, runner, args, embedder)
    if not target_paths:
        return [], []
    source_embeddings = embedder.encode(source_docs, batch_size=getattr(args, "semantic_batch_size", 32))
    similarities = source_embeddings @ target_embeddings.T
    best_source_indices = np.argmax(similarities, axis=0)
    best_scores = np.max(similarities, axis=0)
    ranked = [
        {
            "path": target_path,
            "score": round(float(score), 6),
            "matched_source_path": source_paths[int(source_index)],
            "evidence": {
                "cosine_similarity": round(float(score), 6),
                "retrieval_scope": "all_target_code_files",
                "embedding_model": getattr(args, "embedding_model", DEFAULT_MODEL),
            },
            "baseline": "semantic-file-embedding",
        }
        for target_path, source_index, score in zip(target_paths, best_source_indices, best_scores)
    ]
    ranked.sort(key=lambda item: (-item["score"], item["path"]))
    ranked = ranked[: args.candidate_files]
    statements = map_candidates(runner, state, ranked[: args.inspect_files], args.candidate_statements)
    return ranked, statements[: args.top_statements]


def baseline_codebert(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace, embedder: Embedder) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = path_prefilter(state, args.prefilter_files)
    docs = []
    for item in candidates:
        text = safe_file_at(runner, state["target"].repo, state["target"].base_sha, item["path"])
        docs.append(item["path"] + "\n" + text[: args.max_file_chars])
    query = "\n".join(state["source_changed_paths"]) + "\n" + source_patch_text(state)[: args.max_file_chars]
    sims = embedder.score(query, docs)
    ranked = []
    for item, sim in zip(candidates, sims):
        score = 0.72 * sim + 0.28 * item["score"]
        ranked.append({**item, "score": round(score, 6), "baseline": "CodeBERT-embedding-proxy"})
    ranked = sorted(ranked, key=lambda x: (-x["score"], x["path"]))[: args.candidate_files]
    statements = map_candidates(runner, state, ranked[: args.inspect_files], args.candidate_statements)
    return ranked, statements[: args.top_statements]


def qa_action_ablation(state: dict[str, Any], runner: PropagationQAMCTS, args: argparse.Namespace, actions: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = action_graphqa(state, args.candidate_files)
    if "test" in actions:
        candidates = action_test_impact(state, candidates, args.expanded_files)
    if "symbol" in actions:
        candidates = action_symbol_impact(state, candidates, args.expanded_files)
    if "deletion_interface_build" in actions:
        candidates = action_deletion_interface_build_impact(state, candidates, args.expanded_files)
    if "neighbor" in actions:
        candidates = neighbor_expand(state, candidates, args.expanded_files)
    statements = map_candidates(runner, state, candidates[: args.inspect_files], args.candidate_statements)
    if "round_robin" in actions:
        statements = round_robin_statements(statements, args.top_statements)
    else:
        statements = statements[: args.top_statements]
    return candidates[: args.candidate_files], statements


def result_from(pair: dict[str, Any], index: int, state: dict[str, Any], files: list[dict[str, Any]], statements: list[dict[str, Any]], method: str) -> dict[str, Any]:
    target_files = rank_files_from_evidence(files, statements, 8, True)
    prediction = {
        "propagation_likely": bool(target_files),
        "target_files": target_files,
        "target_statements": statements[:8],
        "strategy": method,
    }
    metrics = evaluate_prediction(state, prediction)
    return {
        "index": index,
        "status": "ok",
        "prediction": prediction,
        "metrics": metrics,
        "ground_truth": {
            "target_changed_files": state["target_changed_paths"],
            "target_changed_old_line_ranges": {
                path: [{"start": start, "end": end} for start, end in ranges]
                for path, ranges in state["target_ranges"].items()
            },
        },
        "method": method,
        "debug": {"source_files": state["source_changed_paths"], "target_files": state["target_changed_paths"]},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    client = GitHubClient(Path(args.cache_dir), os.environ.get("GITHUB_TOKEN"), sleep_seconds=args.sleep)
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    embedder = Embedder(args.embedding_model)
    methods = {
        "Open-NiCad-proxy": [],
        "code2vec-proxy": [],
        "CodeBERT-embedding-proxy": [],
        "QA-GraphOnly": [],
        "QA+TestImpact": [],
        "QA+SymbolImpact": [],
        "QA+DeletionInterfaceBuild": [],
        "QA+AllImpact": [],
    }
    skipped = []
    for idx, pair in enumerate(pairs, start=1):
        runner = PropagationQAMCTS(client, max_nodes=8, top_files=8, top_statements=8)
        state = prepare_state(runner, pair)
        if state.get("status") and state.get("status") != "ok":
            skipped.append({"index": idx, "reason": state.get("reason")})
            continue
        source_summary = runner._execute(state, {"type": "InspectSourcePatch"})
        runner._update_state(state, {"type": "InspectSourcePatch"}, source_summary)

        for method, fn in [
            ("Open-NiCad-proxy", lambda: baseline_open_nicad(state, runner, args)),
            ("code2vec-proxy", lambda: baseline_code2vec(state, runner, args)),
            ("CodeBERT-embedding-proxy", lambda: baseline_codebert(state, runner, args, embedder)),
            ("QA-GraphOnly", lambda: qa_action_ablation(state, runner, args, set())),
            ("QA+TestImpact", lambda: qa_action_ablation(state, runner, args, {"test"})),
            ("QA+SymbolImpact", lambda: qa_action_ablation(state, runner, args, {"symbol"})),
            ("QA+DeletionInterfaceBuild", lambda: qa_action_ablation(state, runner, args, {"deletion_interface_build"})),
            ("QA+AllImpact", lambda: qa_action_ablation(state, runner, args, {"test", "symbol", "deletion_interface_build", "neighbor", "round_robin"})),
        ]:
            files, statements = fn()
            methods[method].append(result_from(pair, idx, state, files, statements, method))

    rows = []
    for method, results in methods.items():
        rows.extend(collect_rows("new_hard_ablation", method, results, set(range(1, len(pairs) + 1)), args.tolerance, datasets=("all",)))
    payload = {
        "summary": {
            "input": args.input,
            "sample_count": len(pairs),
            "skipped": skipped,
            "embedding_model": args.embedding_model,
            "dcard_clone_baseline": "N/A: kevin940726/dcard-clone is a Dcard web app clone, not a clone detector.",
        },
        "rows": rows,
        "results": methods,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/manual_curated_new_hard_samples.json")
    parser.add_argument("--output", default="outputs/new_hard_baseline_action_ablation.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--candidate-files", type=int, default=24)
    parser.add_argument("--expanded-files", type=int, default=48)
    parser.add_argument("--prefilter-files", type=int, default=80)
    parser.add_argument("--inspect-files", type=int, default=16)
    parser.add_argument("--candidate-statements", type=int, default=32)
    parser.add_argument("--top-statements", type=int, default=8)
    parser.add_argument("--max-file-chars", type=int, default=12000)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    payload = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(
        "# New Hard Baseline and QA Action Ablation\n\n"
        + f"- Samples: {payload['summary']['sample_count']}\n"
        + f"- Skipped: {len(payload['summary']['skipped'])}\n"
        + f"- Embedding model used for CodeBERT proxy: `{payload['summary']['embedding_model']}`\n"
        + f"- dcard-clone: {payload['summary']['dcard_clone_baseline']}\n\n"
        + markdown_table("New Hard Samples", payload["rows"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
