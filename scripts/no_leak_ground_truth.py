#!/usr/bin/env python3
"""No-leak ground truth utilities for file/statement propagation detection.

Target PR/commit metadata is never used as detector input. Target changed files
and patches are used only here, as evaluation oracle.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from propagation_detector import GitHubClient, parsed_sides, remote_changed_files, resolve_side, token_similarity


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@dataclass
class StatementGT:
    path: str
    kind: str
    old_start: int
    old_end: int
    new_start: int | None
    new_end: int | None
    old_text: str
    new_text: str
    anchor_line: int
    evaluation_note: str


def meaningful(text: str) -> bool:
    text = (text or "").strip()
    if not text or text in {"{", "}", "},", ");"}:
        return False
    if text.startswith("//") or text.startswith("*"):
        return False
    return len(TOKEN_RE.findall(text)) >= 2


def similarity(a: str, b: str) -> float:
    return 0.55 * SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio() + 0.45 * token_similarity(a, b)


def parse_hunks(file_info: dict[str, Any]) -> list[dict[str, Any]]:
    patch = file_info.get("patch") or ""
    path = file_info.get("filename") or file_info.get("previous_filename") or ""
    hunks: list[dict[str, Any]] = []
    hunk: dict[str, Any] | None = None
    old_line = 0
    new_line = 0
    for raw in patch.splitlines():
        if raw.startswith("@@"):
            if hunk:
                hunks.append(hunk)
            match = re.search(r"-(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?", raw)
            old_line = int(match.group(1)) if match else 0
            new_line = int(match.group(2)) if match else 0
            hunk = {"path": path, "header": raw, "items": []}
            continue
        if hunk is None or raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("-"):
            text = raw[1:].strip()
            if meaningful(text):
                hunk["items"].append({"kind": "deleted", "old_line": old_line, "new_line": None, "text": text})
            old_line += 1
        elif raw.startswith("+"):
            text = raw[1:].strip()
            if meaningful(text):
                hunk["items"].append({"kind": "added", "old_line": None, "new_line": new_line, "text": text, "anchor": max(old_line - 1, 1)})
            new_line += 1
        else:
            text = raw[1:].strip() if raw.startswith(" ") else raw.strip()
            if meaningful(text):
                hunk["items"].append({"kind": "context", "old_line": old_line, "new_line": new_line, "text": text})
            old_line += 1
            new_line += 1
    if hunk:
        hunks.append(hunk)
    return hunks


def pair_deleted_added(deleted: list[dict[str, Any]], added: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    used_added = set()
    for d in deleted:
        best_idx = None
        best_score = 0.0
        for idx, a in enumerate(added):
            if idx in used_added:
                continue
            score = similarity(d["text"], a["text"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= 0.34:
            used_added.add(best_idx)
            pairs.append((d, added[best_idx]))
    paired_deleted = {id(d) for d, _ in pairs}
    remain_deleted = [d for d in deleted if id(d) not in paired_deleted]
    remain_added = [a for idx, a in enumerate(added) if idx not in used_added]
    return pairs, remain_deleted, remain_added


def statement_ground_truth_from_files(files: list[dict[str, Any]]) -> list[StatementGT]:
    out: list[StatementGT] = []
    for file_info in files:
        for hunk in parse_hunks(file_info):
            path = hunk["path"]
            deleted = [x for x in hunk["items"] if x["kind"] == "deleted"]
            added = [x for x in hunk["items"] if x["kind"] == "added"]
            paired, remain_deleted, remain_added = pair_deleted_added(deleted, added)
            for d, a in paired:
                out.append(
                    StatementGT(
                        path=path,
                        kind="edited",
                        old_start=int(d["old_line"]),
                        old_end=int(d["old_line"]),
                        new_start=int(a["new_line"]),
                        new_end=int(a["new_line"]),
                        old_text=d["text"],
                        new_text=a["text"],
                        anchor_line=int(d["old_line"]),
                        evaluation_note="Edited statement: deleted old-line paired with added new-line by text/token similarity.",
                    )
                )
            for d in remain_deleted:
                out.append(
                    StatementGT(
                        path=path,
                        kind="deleted",
                        old_start=int(d["old_line"]),
                        old_end=int(d["old_line"]),
                        new_start=None,
                        new_end=None,
                        old_text=d["text"],
                        new_text="",
                        anchor_line=int(d["old_line"]),
                        evaluation_note="Deleted statement exists in target pre-fix and can be directly localized.",
                    )
                )
            for a in remain_added:
                anchor = int(a.get("anchor") or 1)
                out.append(
                    StatementGT(
                        path=path,
                        kind="added_new",
                        old_start=anchor,
                        old_end=anchor,
                        new_start=int(a["new_line"]),
                        new_end=int(a["new_line"]),
                        old_text="",
                        new_text=a["text"],
                        anchor_line=anchor,
                        evaluation_note="New added statement does not exist pre-fix; evaluate predicted affected locus at insertion anchor/context.",
                    )
                )
    return out


def line_hits_statement_gt(path: str, line: int, gt: StatementGT, tolerance: int) -> bool:
    return path == gt.path and gt.anchor_line - tolerance <= line <= gt.anchor_line + tolerance


def load_pair_target_files(client: GitHubClient, pair: dict[str, Any]) -> list[dict[str, Any]]:
    sides = parsed_sides(pair)
    if sides is None:
        return []
    source, target = (resolve_side(client, side) for side in sides)
    # Pair files are already ordered in curated datasets; target/Infestor is the
    # second side. This is evaluation-only.
    return remote_changed_files(client, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/combined_hard_samples.json")
    parser.add_argument("--output", default="outputs/combined_hard_statement_ground_truth.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    client = GitHubClient(Path(args.cache_dir))
    payload = []
    for index, pair in enumerate(data, start=1):
        files = load_pair_target_files(client, pair)
        gt = statement_ground_truth_from_files(files)
        payload.append(
            {
                "index": index,
                "target_changed_files": [f.get("filename") for f in files if f.get("filename")],
                "statement_ground_truth": [asdict(item) for item in gt],
                "counts": {
                    "edited": sum(1 for item in gt if item.kind == "edited"),
                    "deleted": sum(1 for item in gt if item.kind == "deleted"),
                    "added_new": sum(1 for item in gt if item.kind == "added_new"),
                },
            }
        )
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(payload), "statements": sum(len(x["statement_ground_truth"]) for x in payload)}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
