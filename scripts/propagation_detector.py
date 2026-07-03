#!/usr/bin/env python3
"""Baseline detector for cross-repository blockchain change propagation.

The code is intentionally dependency-free so it can run in a fresh workspace.
It uses GitHub metadata plus lightweight path/content similarity to produce a
first experimental baseline for file-level and statement-level localization.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


PROJECT_TO_REPO = {
    "ethereum_go-ethereum": "ethereum/go-ethereum",
    "bnb-chain_bsc": "bnb-chain/bsc",
    "0xPolygon_bor": "0xPolygon/bor",
    "polygon_bor": "0xPolygon/bor",
}


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


@dataclasses.dataclass(frozen=True)
class Side:
    role: str
    project: str
    repo: str
    kind: str
    number: int | str | None
    resolved_from_kind: str | None
    resolved_from_number: int | None
    title: str
    body: str
    message: str
    url: str
    merged_at: str
    base_sha: str
    merged_sha: str
    file_names: list[str]


@dataclasses.dataclass
class PatchStatement:
    path: str
    line_no: int
    kind: str
    text: str


class GitHubClient:
    def __init__(self, cache_dir: Path, token: str | None = None, sleep_seconds: float = 0.0):
        self.cache_dir = cache_dir
        self.token = token
        self.sleep_seconds = sleep_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str, suffix: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.{suffix}"

    def get_json(self, url: str) -> Any:
        path = self._cache_path(url, "json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        req = urllib.request.Request(url, headers=self._headers("application/vnd.github+json"))
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = resp.read().decode("utf-8")
                break
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        assert payload is not None
        path.write_text(payload, encoding="utf-8")
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return json.loads(payload)

    def get_text(self, url: str) -> str:
        path = self._cache_path(url, "txt")
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        req = urllib.request.Request(url, headers=self._headers("text/plain"))
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = resp.read().decode("utf-8", errors="replace")
                break
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        assert payload is not None
        path.write_text(payload, encoding="utf-8")
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return payload

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "propagation-detector-baseline",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def compare(self, repo: str, base_sha: str, head_sha: str) -> dict[str, Any]:
        base = urllib.parse.quote(base_sha, safe="")
        head = urllib.parse.quote(head_sha, safe="")
        return self.get_json(f"https://api.github.com/repos/{repo}/compare/{base}...{head}")

    def commit(self, repo: str, sha: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(sha, safe="")
        return self.get_json(f"https://api.github.com/repos/{repo}/commits/{quoted}")

    def pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        page = 1
        while True:
            url = f"https://api.github.com/repos/{repo}/pulls/{number}/files?per_page=100&page={page}"
            chunk = self.get_json(url)
            if not chunk:
                break
            files.extend(chunk)
            if len(chunk) < 100:
                break
            page += 1
        return files

    def tree_paths(self, repo: str, sha: str) -> list[str]:
        data = self.get_json(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1")
        return [
            item["path"]
            for item in data.get("tree", [])
            if item.get("type") == "blob" and isinstance(item.get("path"), str)
        ]

    def file_at(self, repo: str, sha: str, path: str) -> str | None:
        git_dir = Path(".cache/git_repos") / f"{repo.replace('/', '_')}.git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", f"--git-dir={git_dir}", "show", f"{sha}:{path}"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout
                # The local mirrors are complete, non-shallow repositories. A
                # failed `git show` therefore means the path is absent at this
                # revision, not that a network fallback is required.
                return None
            except subprocess.TimeoutExpired:
                pass
        quoted = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{quoted}"
        try:
            return self.get_text(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise


def normalize_repo(project: str) -> str:
    if project in PROJECT_TO_REPO:
        return PROJECT_TO_REPO[project]
    if "_" in project:
        owner, name = project.split("_", 1)
        return f"{owner}/{name}"
    return project


def side_from(raw: dict[str, Any], role: str) -> Side | None:
    resolved_by = raw.get("resolved_by") if isinstance(raw.get("resolved_by"), dict) else None
    change = resolved_by or raw
    kind = change.get("type") or raw.get("type") or ""
    base_sha = change.get("base_commit_sha") or ""
    merged_sha = change.get("merged_sha") or change.get("commit_sha") or ""
    merged_at = raw.get("merged_at") or raw.get("closed_at") or ""
    project = raw.get("project") or ""
    if not (project and merged_sha):
        return None
    number = change.get("number")
    if number is None:
        number = change.get("pr_number") or change.get("issue_number")
    if number is None and kind == "commit":
        number = change.get("commit_sha") or change.get("merged_sha")
    return Side(
        role=role,
        project=project,
        repo=normalize_repo(project),
        kind=kind,
        number=number,
        resolved_from_kind=raw.get("type") if resolved_by else None,
        resolved_from_number=raw.get("issue_number") if resolved_by else None,
        title=raw.get("title") or change.get("title") or raw.get("message") or change.get("message") or "",
        body=raw.get("body") or change.get("body") or "",
        message=raw.get("message") or change.get("message") or raw.get("commit_message") or change.get("commit_message") or "",
        url=raw.get("html_url") or "",
        merged_at=merged_at,
        base_sha=base_sha,
        merged_sha=merged_sha,
        file_names=list(change.get("file_names") or raw.get("file_names") or []),
    )


def ordered_sides(pair: dict[str, Any]) -> tuple[Side, Side] | None:
    source = side_from(pair.get("Source", {}), "Source")
    infestor = side_from(pair.get("Infestor", {}), "Infestor")
    if not source or not infestor:
        return None
    if not (source.base_sha and source.merged_at and infestor.base_sha and infestor.merged_at):
        return None
    if source.merged_at <= infestor.merged_at:
        return source, infestor
    return infestor, source


def parsed_sides(pair: dict[str, Any]) -> tuple[Side, Side] | None:
    source = side_from(pair.get("Source", {}), "Source")
    infestor = side_from(pair.get("Infestor", {}), "Infestor")
    if not source or not infestor:
        return None
    return source, infestor


def resolve_side(client: GitHubClient, side: Side) -> Side:
    if side.kind != "commit" or (side.base_sha and side.merged_at):
        return side
    commit_data = client.commit(side.repo, side.merged_sha)
    parents = commit_data.get("parents") or []
    base_sha = side.base_sha
    if not base_sha and parents:
        base_sha = parents[0].get("sha") or ""
    merged_at = side.merged_at
    commit_meta = commit_data.get("commit") or {}
    if not merged_at:
        committer = commit_meta.get("committer") or {}
        author = commit_meta.get("author") or {}
        merged_at = committer.get("date") or author.get("date") or ""
    file_names = side.file_names or [f.get("filename") for f in commit_data.get("files", []) if f.get("filename")]
    return dataclasses.replace(side, base_sha=base_sha, merged_at=merged_at, file_names=file_names)


def order_resolved_sides(source: Side, infestor: Side) -> tuple[Side, Side] | None:
    if not (source.base_sha and source.merged_at and infestor.base_sha and infestor.merged_at):
        return None
    if source.merged_at <= infestor.merged_at:
        return source, infestor
    return infestor, source


def changed_files(compare_data: dict[str, Any], fallback: list[str]) -> list[dict[str, Any]]:
    files = compare_data.get("files") or []
    if files:
        return files
    return [{"filename": name, "patch": ""} for name in fallback]


def remote_changed_files(client: GitHubClient, side: Side) -> list[dict[str, Any]]:
    if side.kind == "pr" and side.number:
        files = client.pull_files(side.repo, side.number)
        if files:
            return files
    if side.kind == "commit":
        commit_data = client.commit(side.repo, side.merged_sha)
        files = commit_data.get("files") or []
        if files:
            return files
    compare_data = client.compare(side.repo, side.base_sha, side.merged_sha)
    return changed_files(compare_data, side.file_names)


def path_tokens(path: str) -> set[str]:
    parts = re.split(r"[/_.\-\s]+", path.lower())
    return {part for part in parts if part}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def path_score(source_path: str, target_path: str) -> float:
    if source_path == target_path:
        return 1.0
    source_name = source_path.rsplit("/", 1)[-1]
    target_name = target_path.rsplit("/", 1)[-1]
    source_ext = source_name.rsplit(".", 1)[-1] if "." in source_name else ""
    target_ext = target_name.rsplit(".", 1)[-1] if "." in target_name else ""
    path_ratio = SequenceMatcher(None, source_path.lower(), target_path.lower()).ratio()
    name_ratio = SequenceMatcher(None, source_name.lower(), target_name.lower()).ratio()
    token_ratio = jaccard(path_tokens(source_path), path_tokens(target_path))
    ext_bonus = 1.0 if source_ext and source_ext == target_ext else 0.0
    return 0.42 * path_ratio + 0.30 * name_ratio + 0.18 * token_ratio + 0.10 * ext_bonus


def rank_files(source_files: Iterable[str], target_tree: list[str], top_k: int) -> list[dict[str, Any]]:
    best_by_target: dict[str, dict[str, Any]] = {}
    for source_path in source_files:
        for target_path in target_tree:
            score = path_score(source_path, target_path)
            old = best_by_target.get(target_path)
            if old is None or score > old["score"]:
                best_by_target[target_path] = {
                    "path": target_path,
                    "score": round(score, 6),
                    "matched_source_path": source_path,
                }
    ranked = sorted(best_by_target.values(), key=lambda item: (-item["score"], item["path"]))
    return ranked[:top_k]


def iter_patch_statements(files: list[dict[str, Any]], include_context: bool = True) -> list[PatchStatement]:
    statements: list[PatchStatement] = []
    hunk_new_line = 0
    hunk_old_line = 0
    for file_info in files:
        path = file_info.get("filename") or file_info.get("previous_filename") or ""
        patch = file_info.get("patch") or ""
        for line in patch.splitlines():
            if line.startswith("@@"):
                match = re.search(r"-(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?", line)
                if match:
                    hunk_old_line = int(match.group(1))
                    hunk_new_line = int(match.group(2))
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                text = line[1:].strip()
                if meaningful_statement(text):
                    statements.append(PatchStatement(path, hunk_new_line, "added", text))
                hunk_new_line += 1
            elif line.startswith("-"):
                text = line[1:].strip()
                if meaningful_statement(text):
                    statements.append(PatchStatement(path, hunk_old_line, "deleted", text))
                hunk_old_line += 1
            else:
                text = line[1:].strip() if line.startswith(" ") else line.strip()
                if include_context and meaningful_statement(text):
                    statements.append(PatchStatement(path, hunk_old_line, "context", text))
                hunk_old_line += 1
                hunk_new_line += 1
    return statements


def meaningful_statement(text: str) -> bool:
    if not text or text in {"{", "}", "},", ");"}:
        return False
    if text.startswith("//") or text.startswith("*"):
        return False
    return len(TOKEN_RE.findall(text)) >= 3


def token_similarity(a: str, b: str) -> float:
    a_tokens = set(TOKEN_RE.findall(a.lower()))
    b_tokens = set(TOKEN_RE.findall(b.lower()))
    return 0.55 * SequenceMatcher(None, a.strip(), b.strip()).ratio() + 0.45 * jaccard(a_tokens, b_tokens)


def rank_statements(
    client: GitHubClient,
    propagator: Side,
    target: Side,
    source_files: list[dict[str, Any]],
    predicted_files: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    source_statements = iter_patch_statements(source_files)
    if not source_statements:
        return []
    ranked: list[dict[str, Any]] = []
    for pred_file in predicted_files:
        target_path = pred_file["path"]
        content = client.file_at(target.repo, target.base_sha, target_path)
        if content is None:
            continue
        lines = content.splitlines()
        comparable_statements = statements_for_file(
            source_statements,
            pred_file.get("matched_source_path", ""),
            target_path,
            max_items=120,
        )
        if not comparable_statements:
            comparable_statements = source_statements[:120]
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not meaningful_statement(stripped):
                continue
            best_statement = max(comparable_statements, key=lambda stmt: token_similarity(stmt.text, stripped))
            sim = token_similarity(best_statement.text, stripped)
            score = 0.70 * sim + 0.30 * float(pred_file["score"])
            ranked.append(
                {
                    "path": target_path,
                    "line": line_no,
                    "score": round(score, 6),
                    "line_text": stripped[:240],
                    "source_path": best_statement.path,
                    "source_line": best_statement.line_no,
                    "source_kind": best_statement.kind,
                    "source_text": best_statement.text[:240],
                }
            )
    ranked.sort(key=lambda item: (-item["score"], item["path"], item["line"]))
    return dedupe_statement_hits(ranked)[:top_k]


def statements_for_file(
    statements: list[PatchStatement],
    source_path: str,
    target_path: str,
    max_items: int,
) -> list[PatchStatement]:
    """Keep statement matching local to the file mapping whenever possible."""
    if not statements:
        return []
    exact = [stmt for stmt in statements if stmt.path == source_path]
    if exact:
        return prioritize_statements(exact)[:max_items]
    source_name = source_path.rsplit("/", 1)[-1]
    target_name = target_path.rsplit("/", 1)[-1]
    same_name = [stmt for stmt in statements if stmt.path.rsplit("/", 1)[-1] in {source_name, target_name}]
    if same_name:
        return prioritize_statements(same_name)[:max_items]
    source_tokens = path_tokens(source_path)
    target_tokens = path_tokens(target_path)
    ranked = sorted(
        statements,
        key=lambda stmt: (
            -max(jaccard(path_tokens(stmt.path), source_tokens), jaccard(path_tokens(stmt.path), target_tokens)),
            kind_priority(stmt.kind),
        ),
    )
    return ranked[:max_items]


def prioritize_statements(statements: list[PatchStatement]) -> list[PatchStatement]:
    return sorted(statements, key=lambda stmt: (kind_priority(stmt.kind), stmt.line_no, stmt.text))


def kind_priority(kind: str) -> int:
    if kind == "deleted":
        return 0
    if kind == "context":
        return 1
    return 2


def dedupe_statement_hits(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = (item["path"], int(item["line"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def changed_old_line_ranges(files: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = {}
    for file_info in files:
        path = file_info.get("filename") or ""
        patch = file_info.get("patch") or ""
        old_line = 0
        start: int | None = None
        end: int | None = None
        for line in patch.splitlines():
            if line.startswith("@@"):
                if start is not None and end is not None:
                    ranges.setdefault(path, []).append((start, end))
                start = None
                end = None
                match = re.search(r"-(\d+)(?:,(\d+))? \+\d+(?:,\d+)?", line)
                old_line = int(match.group(1)) if match else 0
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                if start is None:
                    start = max(old_line - 1, 1)
                end = max(old_line, start)
            elif line.startswith("-"):
                if start is None:
                    start = old_line
                end = old_line
                old_line += 1
            else:
                old_line += 1
        if start is not None and end is not None:
            ranges.setdefault(path, []).append((start, end))
    return ranges


def line_hits_ranges(path: str, line: int, ranges: dict[str, list[tuple[int, int]]], tolerance: int) -> bool:
    return any(start - tolerance <= line <= end + tolerance for start, end in ranges.get(path, []))


class PropagationExperiment:
    def __init__(self, client: GitHubClient, top_files: int, top_statements: int, line_tolerance: int):
        self.client = client
        self.top_files = top_files
        self.top_statements = top_statements
        self.line_tolerance = line_tolerance

    def run_pair(self, index: int, pair: dict[str, Any]) -> dict[str, Any]:
        sides = parsed_sides(pair)
        if sides is None:
            return {"index": index, "status": "skipped", "reason": "missing commit or merge metadata"}
        source_side, infestor_side = (resolve_side(self.client, side) for side in sides)
        ordered = order_resolved_sides(source_side, infestor_side)
        if ordered is None:
            return {"index": index, "status": "skipped", "reason": "missing base commit or date metadata"}
        propagator, target = ordered
        source_files = remote_changed_files(self.client, propagator)
        target_files = remote_changed_files(self.client, target)
        source_changed_paths = [f.get("filename") for f in source_files if f.get("filename")]
        target_changed_paths = [f.get("filename") for f in target_files if f.get("filename")]
        target_tree = self.client.tree_paths(target.repo, target.base_sha)

        predicted_files = rank_files(source_changed_paths or propagator.file_names, target_tree, self.top_files)
        predicted_statements = rank_statements(
            self.client,
            propagator,
            target,
            source_files,
            predicted_files,
            self.top_statements,
        )
        target_ranges = changed_old_line_ranges(target_files)
        target_changed_set = set(target_changed_paths)
        file_hits = [item for item in predicted_files if item["path"] in target_changed_set]
        statement_hits = [
            item
            for item in predicted_statements
            if line_hits_ranges(item["path"], int(item["line"]), target_ranges, self.line_tolerance)
        ]
        return {
            "index": index,
            "status": "ok",
            "propagator": dataclasses.asdict(propagator),
            "target": dataclasses.asdict(target),
            "source_changed_files": source_changed_paths,
            "target_changed_files": target_changed_paths,
            "predicted_files": predicted_files,
            "predicted_statements": predicted_statements,
            "metrics": {
                "file_topk_hit": bool(file_hits),
                "file_hits": file_hits,
                "statement_topk_hit": bool(statement_hits),
                "statement_hits": statement_hits,
                "target_changed_old_line_ranges": {
                    path: [{"start": start, "end": end} for start, end in ranges]
                    for path, ranges in target_ranges.items()
                },
            },
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [result for result in results if result.get("status") == "ok"]
    skipped = [result for result in results if result.get("status") != "ok"]
    file_hits = sum(1 for result in ok if result.get("metrics", {}).get("file_topk_hit"))
    statement_hits = sum(1 for result in ok if result.get("metrics", {}).get("statement_topk_hit"))
    return {
        "evaluated_pairs": len(ok),
        "skipped_pairs": len(skipped),
        "file_topk_hits": file_hits,
        "file_topk_hit_rate": round(file_hits / len(ok), 4) if ok else None,
        "statement_topk_hits": statement_hits,
        "statement_topk_hit_rate": round(statement_hits / len(ok), 4) if ok else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="sample_30_pairs.json", help="Input pair JSON file.")
    parser.add_argument("--output", default="outputs/propagation_results.json", help="Output JSON file.")
    parser.add_argument("--max-pairs", type=int, default=20, help="Maximum number of pairs to process.")
    parser.add_argument("--top-files", type=int, default=5, help="Number of target file candidates per pair.")
    parser.add_argument("--top-statements", type=int, default=10, help="Number of statement candidates per pair.")
    parser.add_argument("--line-tolerance", type=int, default=5, help="Statement hit tolerance around target patch lines.")
    parser.add_argument("--cache-dir", default=".cache/github", help="GitHub response cache directory.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay after uncached GitHub requests.")
    parser.add_argument("--indices", default="", help="Comma-separated 1-based input indices to run.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write output after this many processed pairs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    selected_indices = {
        int(part.strip())
        for part in args.indices.split(",")
        if part.strip().isdigit() and int(part.strip()) > 0
    }
    run_items = [
        (index, pair)
        for index, pair in enumerate(pairs, start=1)
        if not selected_indices or index in selected_indices
    ]
    if args.max_pairs:
        run_items = run_items[: args.max_pairs]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"), sleep_seconds=args.sleep)
    experiment = PropagationExperiment(client, args.top_files, args.top_statements, args.line_tolerance)

    results: list[dict[str, Any]] = []
    output_path = Path(args.output)
    for offset, (index, pair) in enumerate(run_items, start=1):
        try:
            result = experiment.run_pair(index, pair)
        except Exception as exc:
            result = {"index": index, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(f"[{offset}/{len(run_items)}; input_index={index}] {result['status']}: {result.get('reason', '')}", flush=True)
        if args.checkpoint_every and offset % args.checkpoint_every == 0:
            payload = {
                "summary": summarize(results),
                "results": results,
                "errors": [result for result in results if result.get("status") != "ok"],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "summary": summarize(results),
        "results": results,
        "errors": [result for result in results if result.get("status") != "ok"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
