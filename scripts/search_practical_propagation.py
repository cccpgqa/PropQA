#!/usr/bin/env python3
"""Search repository history for real-world cross-client change propagation.

The search starts from upstream pull requests, localizes their likely impact in
target repositories at the source merge time, and then inspects later target
commits for changes at the localized files and statements. The output is a
ranked review queue; a candidate becomes a validated propagation case only
after manual inspection confirms that both changes address the same issue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from propagation_detector import (
    GitHubClient,
    Side,
    iter_patch_statements,
    remote_changed_files,
    token_similarity,
)
from rerun_file_llm_v3 import (
    candidate_code_preview,
    candidate_symbols,
    compact_source_statements,
    rerank_batch,
)
from rerun_statement_existing_code_v3 import LLMClient, deletion_side_blocks, llm_rerank_statements
from run_graphqa_action_enhancement_experiment import expand_direct_blocks
from run_file_pair_detection_experiment import restrict_state_to_source_file
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv
from run_rq5_action_ablation_v3 import file_action_rank


DEFAULT_TARGETS = ("bnb-chain/bsc", "0xPolygon/bor", "celo-org/celo-blockchain")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|\d+")


def api_url(path: str, **params: str | int) -> str:
    query = urllib.parse.urlencode(params)
    return f"https://api.github.com/{path}?{query}" if query else f"https://api.github.com/{path}"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def text_tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "from", "with", "this", "that", "geth", "bsc", "bor"}
    return {token.lower() for token in WORD_RE.findall(text) if token.lower() not in stop}


def jaccard_text(left: str, right: str) -> float:
    a, b = text_tokens(left), text_tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def patch_lines(files: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    lines = {"added": [], "deleted": []}
    for item in files:
        for line in str(item.get("patch") or "").splitlines():
            if line.startswith(("+++", "---", "@@")):
                continue
            text = line[1:].strip() if line[:1] in {"+", "-"} else ""
            if len(text) < 4:
                continue
            if line.startswith("+"):
                lines["added"].append(text)
            elif line.startswith("-"):
                lines["deleted"].append(text)
    return lines


def directional_patch_similarity(
    source_files: Iterable[dict[str, Any]], target_files: Iterable[dict[str, Any]]
) -> float:
    source, target = patch_lines(source_files), patch_lines(target_files)
    scores = []
    for operation in ("added", "deleted"):
        for line in source[operation][:80]:
            if target[operation]:
                scores.append(max(token_similarity(line, other) for other in target[operation][:120]))
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return sum(scores[: min(20, len(scores))]) / min(20, len(scores))


def statement_evidence(
    predicted: list[dict[str, Any]], target_files: list[dict[str, Any]], threshold: float = 0.68
) -> list[dict[str, Any]]:
    removed = patch_lines(target_files)["deleted"]
    hits = []
    for item in predicted:
        source_text = str(item.get("line_text") or item.get("text") or "").strip()
        if not source_text or not removed:
            continue
        matched = max(removed, key=lambda line: token_similarity(source_text, line))
        score = token_similarity(source_text, matched)
        if score >= threshold:
            hits.append(
                {
                    "predicted_path": item.get("path"),
                    "predicted_line": item.get("line"),
                    "predicted_statement": source_text[:300],
                    "later_removed_statement": matched[:300],
                    "similarity": round(score, 4),
                }
            )
    return hits


def score_later_change(
    source: Side,
    source_files: list[dict[str, Any]],
    predicted_files: list[str],
    predicted_statements: list[dict[str, Any]],
    commit: dict[str, Any],
) -> dict[str, Any]:
    target_files = commit.get("files") or []
    changed = [item.get("filename") for item in target_files if item.get("filename")]
    overlap = [path for path in predicted_files if path in set(changed)]
    file_score = len(overlap) / max(1, min(len(predicted_files), len(changed)))
    patch_score = directional_patch_similarity(source_files, target_files)
    message = ((commit.get("commit") or {}).get("message") or "").splitlines()[0]
    semantic_score = jaccard_text(" ".join((source.title, source.body, source.message)), message)
    statement_hits = statement_evidence(predicted_statements, target_files)
    statement_score = min(1.0, len(statement_hits) / max(1, len(predicted_statements)))
    total = 0.38 * file_score + 0.37 * patch_score + 0.15 * semantic_score + 0.10 * statement_score
    return {
        "score": round(total, 6),
        "file_score": round(file_score, 6),
        "patch_score": round(patch_score, 6),
        "semantic_score": round(semantic_score, 6),
        "statement_score": round(statement_score, 6),
        "matched_files": overlap,
        "changed_files": changed,
        "statement_evidence": statement_hits,
        "message": message,
    }


def merged_source_prs(
    client: GitHubClient, repo: str, since: str, until: str, limit: int
) -> list[dict[str, Any]]:
    query = f"repo:{repo} is:pr is:merged merged:{since[:10]}..{until[:10]}"
    pulls = []
    page = 1
    while len(pulls) < limit:
        page_size = min(100, limit - len(pulls))
        search = client.get_json(
            api_url(
                "search/issues",
                q=query,
                sort="created",
                order="asc",
                per_page=page_size,
                page=page,
            )
        )
        items = search.get("items") or []
        if not items:
            break
        for item in items:
            pulls.append(
                client.get_json(f"https://api.github.com/repos/{repo}/pulls/{item['number']}")
            )
        if len(items) < page_size:
            break
        page += 1
    return pulls


def load_source_prs(client: GitHubClient, path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("sources", [])
    pulls = []
    for item in items:
        if item.get("merged_at") and item.get("merge_commit_sha"):
            pulls.append(item)
            continue
        repo = item["repo"]
        number = int(item["number"])
        pulls.append(client.get_json(f"https://api.github.com/repos/{repo}/pulls/{number}"))
    return pulls


def source_side(repo: str, pull: dict[str, Any]) -> Side:
    repo = str((((pull.get("base") or {}).get("repo") or {}).get("full_name")) or repo)
    return Side(
        role="source",
        project=repo.replace("/", "_"),
        repo=repo,
        kind="pr",
        number=int(pull["number"]),
        resolved_from_kind=None,
        resolved_from_number=None,
        title=str(pull.get("title") or ""),
        body=str(pull.get("body") or ""),
        message="",
        url=str(pull.get("html_url") or ""),
        merged_at=str(pull.get("merged_at") or ""),
        base_sha=str((pull.get("base") or {}).get("sha") or ""),
        merged_sha=str(pull.get("merge_commit_sha") or ""),
        file_names=[],
    )


def repository_snapshot(client: GitHubClient, repo: str, at: str) -> tuple[str, str]:
    commits = client.get_json(api_url(f"repos/{repo}/commits", until=at, per_page=1))
    if not commits:
        raise RuntimeError(f"no {repo} revision available at {at}")
    item = commits[0]
    committed = ((item.get("commit") or {}).get("committer") or {}).get("date") or at
    return str(item["sha"]), str(committed)


def target_side(repo: str, sha: str, at: str) -> Side:
    return Side(
        role="target",
        project=repo.replace("/", "_"),
        repo=repo,
        kind="snapshot",
        number=None,
        resolved_from_kind=None,
        resolved_from_number=None,
        title="",
        body="",
        message="",
        url=f"https://github.com/{repo}/tree/{sha}",
        merged_at=at,
        base_sha=sha,
        merged_sha=sha,
        file_names=[],
    )


def build_localization_state(
    client: GitHubClient, source: Side, target: Side
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_files = remote_changed_files(client, source)
    source_paths = [item.get("filename") for item in source_files if item.get("filename")]
    state = {
        "status": "ok",
        "propagator": source,
        "target": target,
        "source_files": source_files,
        "source_changed_paths": source_paths,
        "source_statements": iter_patch_statements(source_files, include_context=True),
        "target_tree": client.tree_paths(target.repo, target.base_sha),
        "source_symbol_cache": {},
        "target_symbol_cache": {},
    }
    return state, source_files


def localize(
    runner: PropagationQAMCTS,
    llm: LLMClient,
    state: dict[str, Any],
    batch_size: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    queries = []
    graph_candidates: dict[str, list[dict[str, Any]]] = {}
    for source_file in state["source_changed_paths"]:
        one_file_state = restrict_state_to_source_file(state, source_file)
        candidates = file_action_rank(one_file_state, 20)
        for item in candidates[:10]:
            item["target_symbols"] = candidate_symbols(runner, one_file_state, item["path"])
            item["target_code_preview"] = candidate_code_preview(
                runner, one_file_state, item["path"], item["target_symbols"]
            )
        graph_candidates[source_file] = candidates
        queries.append(
            {
                "source_file": source_file,
                "source_statements": compact_source_statements(state, source_file),
                "candidates": candidates,
            }
        )
    ranked_by_source: dict[str, list[str]] = {}
    traces = []
    for start in range(0, len(queries), batch_size):
        ranked, trace = rerank_batch(llm, state, queries[start : start + batch_size])
        ranked_by_source.update(ranked)
        traces.append(trace)
    ranked_files = []
    for source_file in state["source_changed_paths"]:
        paths = ranked_by_source.get(source_file) or [item["path"] for item in graph_candidates[source_file]][:5]
        for path in paths:
            if path not in ranked_files:
                ranked_files.append(path)
    ranked_files = ranked_files[:20]
    blocks, intents = deletion_side_blocks(state, runner, ranked_files)
    budget = min(1200, max(100, 8 * len(intents), 6 * len(ranked_files)))
    statement_candidates = expand_direct_blocks(
        runner.client,
        state,
        blocks,
        intents,
        budget,
        6,
        use_insert_anchor=False,
    )
    ranked_statements, statement_trace = llm_rerank_statements(
        runner.client, llm, state, blocks, intents, statement_candidates, 100
    )
    traces.append({"statement_localization": statement_trace})
    return ranked_files, ranked_statements, traces


def commits_touching_predictions(
    client: GitHubClient,
    repo: str,
    paths: list[str],
    since: str,
    until: str,
    per_path: int,
) -> list[dict[str, Any]]:
    commits: dict[str, dict[str, Any]] = {}
    for path in paths:
        items = client.get_json(
            api_url(f"repos/{repo}/commits", path=path, since=since, until=until, per_page=per_path)
        )
        for item in items:
            commits.setdefault(str(item["sha"]), item)
    return list(commits.values())


def search_one_target(
    client: GitHubClient,
    runner: PropagationQAMCTS,
    llm: LLMClient,
    source: Side,
    repo: str,
    until: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    snapshot_sha, snapshot_time = repository_snapshot(client, repo, source.merged_at)
    target = target_side(repo, snapshot_sha, snapshot_time)
    state, source_files = build_localization_state(client, source, target)
    files, statements, traces = localize(runner, llm, state, args.batch_queries)
    history = commits_touching_predictions(
        client, repo, files, source.merged_at, until, args.commits_per_file
    )
    candidates = []
    for item in history:
        commit = client.commit(repo, str(item["sha"]))
        evidence = score_later_change(source, source_files, files, statements, commit)
        if evidence["score"] < args.min_score:
            continue
        candidates.append(
            {
                "target_sha": commit["sha"],
                "target_url": commit.get("html_url") or f"https://github.com/{repo}/commit/{commit['sha']}",
                "target_date": ((commit.get("commit") or {}).get("committer") or {}).get("date"),
                **evidence,
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item.get("target_date") or ""))
    return {
        "target_repo": repo,
        "snapshot_sha": snapshot_sha,
        "snapshot_time": snapshot_time,
        "predicted_files": files,
        "predicted_statements": statements,
        "qa_traces": traces,
        "history_candidates": candidates[: args.max_candidates],
    }


def smoke_test() -> None:
    source = Side(
        "source", "ethereum_go-ethereum", "ethereum/go-ethereum", "pr", 1,
        None, None, "fix batch lifecycle", "close batch after write", "",
        "https://example/source", "2024-01-01T00:00:00Z", "base", "head", ["core/db.go"],
    )
    source_files = [{"filename": "core/db.go", "patch": "@@ -1 +1,2 @@\n batch := db.NewBatch()\n+defer batch.Close()"}]
    similar = {
        "sha": "same",
        "commit": {"message": "fix batch lifecycle", "committer": {"date": "2024-02-01T00:00:00Z"}},
        "files": [{"filename": "core/db.go", "patch": "@@ -1 +1,2 @@\n batch := db.NewBatch()\n+defer batch.Close()"}],
    }
    unrelated = {
        "sha": "other",
        "commit": {"message": "update documentation", "committer": {"date": "2024-02-01T00:00:00Z"}},
        "files": [{"filename": "docs/readme.md", "patch": "@@ -1 +1 @@\n-old\n+new"}],
    }
    statements = [{"path": "core/db.go", "line": 1, "text": "batch := db.NewBatch()"}]
    a = score_later_change(source, source_files, ["core/db.go"], statements, similar)
    b = score_later_change(source, source_files, ["core/db.go"], statements, unrelated)
    assert a["score"] > b["score"]
    assert a["matched_files"] == ["core/db.go"]
    assert directional_patch_similarity(source_files, similar["files"]) > 0.9
    print("Practical-history search smoke test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default="ethereum/go-ethereum")
    parser.add_argument("--source-changes", type=Path)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--since", default="2024-01-01T00:00:00Z")
    parser.add_argument("--until", default=iso_time(datetime.now(timezone.utc)))
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--max-sources", type=int, default=100)
    parser.add_argument("--commits-per-file", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--batch-queries", type=int, default=5)
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--output", default="results/practical_history_candidates.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return 0
    load_dotenv()
    llm = LLMClient("PropQA", os.environ.get("API_KEY", ""), os.environ.get("BASE_URL", ""), os.environ.get("MODEL", ""))
    if not llm.api_key or not llm.base_url or not llm.model:
        raise RuntimeError("API_KEY, BASE_URL, and MODEL must be configured")
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    pulls = (
        load_source_prs(client, args.source_changes)
        if args.source_changes
        else merged_source_prs(client, args.source_repo, args.since, args.until, args.max_sources)
    )
    targets = [repo.strip() for repo in args.targets.split(",") if repo.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        results = list(previous.get("results") or [])
        errors = list(previous.get("errors") or [])
    else:
        results, errors = [], []
    completed_sources = {item.get("source_url") for item in results}

    def write_checkpoint() -> None:
        payload = {
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "smoke_test"
            },
            "validation_note": (
                "Candidates require manual confirmation that source and target changes "
                "address the same software issue."
            ),
            "sources_searched": len(results),
            "results": results,
            "errors": errors,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for pull in pulls[: args.max_sources]:
        source = source_side(args.source_repo, pull)
        if not source.merged_at or not source.merged_sha:
            continue
        if source.url in completed_sources:
            continue
        history_until = min(parse_time(args.until), parse_time(source.merged_at) + timedelta(days=args.history_days))
        item = {"source_repo": source.repo, "source_pr": source.number, "source_url": source.url, "source_merged_at": source.merged_at, "targets": []}
        for repo in targets:
            if repo == source.repo:
                continue
            try:
                item["targets"].append(search_one_target(client, runner, llm, source, repo, iso_time(history_until), args))
            except Exception as exc:
                errors.append({"source_url": source.url, "target_repo": repo, "error": f"{type(exc).__name__}: {exc}"})
        results.append(item)
        write_checkpoint()
        print(f"searched {source.url}: {len(item['targets'])} target repositories", flush=True)
    write_checkpoint()
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
