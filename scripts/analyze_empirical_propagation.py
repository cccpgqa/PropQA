#!/usr/bin/env python3
"""Empirical analysis for high-precision propagation pairs.

The script uses only the Python standard library and writes publication-friendly
SVG figures plus Markdown/JSON/CSV analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


REPO_LABELS = {
    "ethereum_go-ethereum": "Ethereum",
    "bnb-chain_bsc": "BSC",
    "0xPolygon_bor": "Polygon",
    "polygon_bor": "Polygon",
    "ethereum/go-ethereum": "Ethereum",
    "bnb-chain/bsc": "BSC",
    "0xPolygon/bor": "Polygon",
}
MAIN_REPOS = ["Ethereum", "BSC", "Polygon"]
TYPE_ORDER = ["bug_fix", "new_feature", "refactor", "dependency_security", "upstream_sync", "test_doc", "maintenance"]
TYPE_LABELS = {
    "bug_fix": "Bug fix",
    "new_feature": "New feature",
    "refactor": "Refactor",
    "dependency_security": "Dependency/Security",
    "upstream_sync": "Upstream sync",
    "test_doc": "Test/Doc",
    "maintenance": "Maintenance",
}
TYPE_COLORS = {
    "bug_fix": "#D95F5F",
    "new_feature": "#3B7EA1",
    "refactor": "#7A6BBC",
    "dependency_security": "#C77C2E",
    "upstream_sync": "#5C8A4A",
    "test_doc": "#8C8C8C",
    "maintenance": "#4A9C8F",
}
DIRECTION_COLORS = {
    "Ethereum->BSC": "#2F6FAD",
    "Ethereum->Polygon": "#6C5FA7",
    "BSC->Ethereum": "#C65D4B",
    "BSC->Polygon": "#D08B32",
    "Polygon->Ethereum": "#3E8B67",
    "Polygon->BSC": "#86A83E",
}


def parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def repo_label(project: str) -> str:
    return REPO_LABELS.get(project, project.replace("_", "/"))


def side_change(side: dict[str, Any]) -> dict[str, Any]:
    resolved = side.get("resolved_by") if isinstance(side.get("resolved_by"), dict) else None
    return resolved or side


def side_date(side: dict[str, Any]) -> datetime | None:
    change = side_change(side)
    # Propagation direction follows when a change first becomes observable.
    # Merge order is analyzed separately because review latency can reverse it.
    for key in ("created_at", "date", "merged_at", "closed_at", "updated_at"):
        dt = parse_dt(change.get(key) or side.get(key) or "")
        if dt:
            return dt
    return None


def side_text(side: dict[str, Any]) -> str:
    change = side_change(side)
    return " ".join(
        [
            side.get("title", ""),
            side.get("body", ""),
            side.get("message", ""),
            change.get("title", ""),
            change.get("body", ""),
            change.get("message", ""),
        ]
    )


def side_title(side: dict[str, Any]) -> str:
    change = side_change(side)
    title = side.get("title") or change.get("title")
    if title:
        return title
    message = side.get("message") or change.get("message") or ""
    return message.splitlines()[0] if message else ""


def side_files(side: dict[str, Any]) -> list[str]:
    change = side_change(side)
    return list(change.get("file_names") or side.get("file_names") or [])


def change_id(side: dict[str, Any]) -> str:
    change = side_change(side)
    kind = change.get("type") or side.get("type")
    if kind == "pr":
        return f"PR #{change.get('pr_number') or change.get('number') or side.get('pr_number')}"
    sha = change.get("commit_sha") or change.get("merged_sha") or side.get("commit_sha") or side.get("merged_sha") or ""
    return sha[:8] if sha else "unknown"


def classify_change(source: dict[str, Any], target: dict[str, Any]) -> str:
    text = " ".join([side_text(source), side_text(target)]).lower()
    if re.search(r"\bcve-\d{4}-\d+|security|vulnerab|dos|ddos|panic|crash|deadlock|race|overflow|invalid|error|bug|fix|regression|flaky", text):
        return "bug_fix"
    if re.search(r"\bupstream|merge tag|geth-v|sync geth|merge geth|cherry[- ]pick", text):
        return "upstream_sync"
    if re.search(r"\bgo\.mod|go\.sum|bump|upgrade|dependency|dependencies|update .*version|crypto to solve", text):
        return "dependency_security"
    if re.search(r"\brefactor|cleanup|clean up|rename|restructure|rework|simplify", text):
        return "refactor"
    if re.search(r"\btest|doc|readme|comment|example|fixture", text):
        return "test_doc"
    if re.search(r"\badd|implement|support|introduce|enable|new |allow|expose", text):
        return "new_feature"
    return "maintenance"


def earlier_later(pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    source = pair.get("Source") or {}
    target = pair.get("Infestor") or {}
    s_dt = side_date(source)
    t_dt = side_date(target)
    if not s_dt or not t_dt:
        return None
    return (source, target) if s_dt <= t_dt else (target, source)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    pos = (len(data) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)


def fmt_days(value: float) -> str:
    if value < 1:
        return f"{value * 24:.1f}h"
    if value < 30:
        return f"{value:.1f}d"
    return f"{value:.0f}d"


def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
        ".title{font-size:22px;font-weight:700}",
        ".subtitle{font-size:13px;fill:#52616b}",
        ".axis{font-size:12px;fill:#52616b}",
        ".label{font-size:13px}",
        ".small{font-size:11px;fill:#52616b}",
        ".grid{stroke:#d9e2ec;stroke-width:1}",
        ".axisline{stroke:#9fb3c8;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join(parts + ["</svg>\n"]), encoding="utf-8")


def draw_direction_matrix(records: list[dict[str, Any]], out: Path) -> None:
    counts = Counter(r["direction"] for r in records)
    max_count = max(counts.values()) if counts else 1
    repo_counts = Counter()
    for r in records:
        repo_counts[r["source_repo"]] += 1
        repo_counts[r["target_repo"]] += 1
    repos = [repo for repo, _ in repo_counts.most_common(6)]
    width, height = 900, 240 + 86 * len(repos)
    left, top, cell, gap = 220, 120, 76, 12
    parts = svg_header(width, height)
    parts += [
        '<text class="title" x="40" y="42">Propagation Direction Matrix</text>',
        '<text class="subtitle" x="40" y="66">Rows are propagators; columns are adopters. Cell intensity encodes pair count.</text>',
    ]
    for j, col in enumerate(repos):
        x = left + j * (cell + gap) + cell / 2
        parts.append(f'<text class="label" x="{x:.1f}" y="{top-22}" text-anchor="middle">{esc(col)}</text>')
    for i, row in enumerate(repos):
        y = top + i * (cell + gap) + cell / 2 + 5
        parts.append(f'<text class="label" x="{left-26}" y="{y:.1f}" text-anchor="end">{esc(row)}</text>')
        for j, col in enumerate(repos):
            direction = f"{row}->{col}"
            x = left + j * (cell + gap)
            yy = top + i * (cell + gap)
            if row == col:
                fill = "#f5f7fa"
                text = "—"
            else:
                count = counts.get(direction, 0)
                alpha = 0.14 + 0.76 * (count / max_count if max_count else 0)
                base = DIRECTION_COLORS.get(direction, "#607D8B")
                fill = base
                text = str(count)
                parts.append(f'<rect x="{x}" y="{yy}" width="{cell}" height="{cell}" rx="8" fill="{fill}" opacity="{alpha:.3f}"/>')
                parts.append(f'<text class="label" x="{x+cell/2}" y="{yy+cell/2+6}" text-anchor="middle" font-weight="700">{text}</text>')
                continue
            parts.append(f'<rect x="{x}" y="{yy}" width="{cell}" height="{cell}" rx="8" fill="{fill}" stroke="#d9e2ec"/>')
            parts.append(f'<text class="small" x="{x+cell/2}" y="{yy+cell/2+4}" text-anchor="middle">{text}</text>')
    ranked = counts.most_common()
    x0, y0 = left + len(repos) * (cell + gap) + 40, 120
    parts.append(f'<text class="label" x="{x0}" y="{y0}" font-weight="700">Top directions</text>')
    for idx, (direction, count) in enumerate(ranked[:6], start=1):
        y = y0 + 28 * idx
        color = DIRECTION_COLORS.get(direction, "#607D8B")
        parts.append(f'<circle cx="{x0+8}" cy="{y-4}" r="6" fill="{color}"/>')
        parts.append(f'<text class="axis" x="{x0+24}" y="{y}">{esc(direction)}: {count}</text>')
    write_svg(out, parts)


def draw_latency_histogram(records: list[dict[str, Any]], out: Path) -> None:
    bins = [
        ("<=1d", 0, 1),
        ("1-7d", 1, 7),
        ("7-30d", 7, 30),
        ("30-90d", 30, 90),
        ("90-180d", 90, 180),
        (">180d", 180, float("inf")),
    ]
    counts = []
    for _, lo, hi in bins:
        counts.append(sum(1 for r in records if lo <= r["delay_days"] < hi))
    width, height = 820, 520
    left, right, top, bottom = 90, 40, 90, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    max_count = max(counts) if counts else 1
    parts = svg_header(width, height)
    delays = [r["delay_days"] for r in records]
    nonzero = [d for d in delays if d > 0]
    delay_ref = nonzero or delays
    parts += [
        '<text class="title" x="40" y="42">Propagation Latency Distribution</text>',
        f'<text class="subtitle" x="40" y="66">All pairs are shown. Non-zero latency: median={fmt_days(median(delay_ref))}, P75={fmt_days(quantile(delay_ref, 0.75))}, zero-lag imported commits={len(delays)-len(nonzero)}.</text>',
    ]
    for t in range(0, max_count + 1, max(1, math.ceil(max_count / 5))):
        y = top + plot_h - (t / max_count) * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{t}</text>')
    bar_gap = 18
    bar_w = (plot_w - bar_gap * (len(bins) - 1)) / len(bins)
    for i, ((label, _, _), count) in enumerate(zip(bins, counts)):
        x = left + i * (bar_w + bar_gap)
        h = (count / max_count) * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="#3B7EA1"/>')
        parts.append(f'<text class="label" x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle">{count}</text>')
        parts.append(f'<text class="axis" x="{x+bar_w/2:.1f}" y="{height-50}" text-anchor="middle">{esc(label)}</text>')
    parts.append(f'<line class="axisline" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}"/>')
    parts.append(f'<text class="axis" x="{width/2}" y="{height-18}" text-anchor="middle">Merged-to-merged latency</text>')
    parts.append(f'<text class="axis" transform="translate(24 {height/2}) rotate(-90)" text-anchor="middle">Number of pairs</text>')
    write_svg(out, parts)


def x_log(value: float, max_day: float, left: int, plot_w: int) -> float:
    return left + (math.log10(value + 1) / math.log10(max_day + 1)) * plot_w


def draw_latency_by_direction(records: list[dict[str, Any]], out: Path) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["delay_days"] > 0:
            groups[r["direction"]].append(r["delay_days"])
    directions = [d for d, _ in Counter(r["direction"] for r in records).most_common()]
    width, height = 940, 120 + 58 * len(directions)
    left, right, top, row_h = 210, 60, 90, 58
    plot_w = width - left - right
    max_day = max(max(vals) for vals in groups.values())
    ticks = [0, 1, 7, 30, 90, 180, 365, 730, 1095]
    parts = svg_header(width, height)
    parts += [
        '<text class="title" x="40" y="42">Latency by Propagation Direction</text>',
        '<text class="subtitle" x="40" y="66">Only non-zero, observable latencies are used. Box shows IQR; whiskers show P10-P90.</text>',
    ]
    y_axis = top + row_h * len(directions) + 8
    for tick in ticks:
        if tick > max_day:
            continue
        x = x_log(tick, max_day, left, plot_w)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top-20}" y2="{y_axis}"/>')
        parts.append(f'<text class="axis" x="{x:.1f}" y="{y_axis+24}" text-anchor="middle">{tick}</text>')
    for i, direction in enumerate(directions):
        vals = groups[direction]
        y = top + i * row_h
        q10, q25, q50, q75, q90 = [quantile(vals, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
        x10, x25, x50, x75, x90 = [x_log(v, max_day, left, plot_w) for v in (q10, q25, q50, q75, q90)]
        color = DIRECTION_COLORS.get(direction, "#607D8B")
        parts.append(f'<text class="label" x="{left-18}" y="{y+6}" text-anchor="end">{esc(direction)} ({len(vals)})</text>')
        parts.append(f'<line x1="{x10:.1f}" x2="{x90:.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3" opacity="0.75"/>')
        parts.append(f'<rect x="{x25:.1f}" y="{y-13}" width="{max(2, x75-x25):.1f}" height="26" rx="5" fill="{color}" opacity="0.28" stroke="{color}"/>')
        parts.append(f'<line x1="{x50:.1f}" x2="{x50:.1f}" y1="{y-16}" y2="{y+16}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{x50+8:.1f}" y="{y-18}">median {fmt_days(q50)}</text>')
    parts.append(f'<text class="axis" x="{left+plot_w/2}" y="{height-22}" text-anchor="middle">Latency in days after propagator merge</text>')
    write_svg(out, parts)


def draw_type_stacked(records: list[dict[str, Any]], out: Path) -> None:
    directions = [d for d, _ in Counter(r["direction"] for r in records).most_common()]
    counts: dict[str, Counter[str]] = {d: Counter(r["change_type"] for r in records if r["direction"] == d) for d in directions}
    totals = {d: sum(counts[d].values()) for d in directions}
    width, height = 980, 170 + 48 * len(directions)
    left, right, top, row_h = 210, 250, 100, 48
    plot_w = width - left - right
    parts = svg_header(width, height)
    parts += [
        '<text class="title" x="40" y="42">Change Type Composition by Direction</text>',
        '<text class="subtitle" x="40" y="66">Types are keyword-based and intended for empirical characterization.</text>',
    ]
    for i, d in enumerate(directions):
        y = top + i * row_h
        x = left
        parts.append(f'<text class="label" x="{left-16}" y="{y+17}" text-anchor="end">{esc(d)} ({totals[d]})</text>')
        for typ in TYPE_ORDER:
            value = counts[d].get(typ, 0)
            if not value:
                continue
            w = plot_w * value / totals[d]
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="26" rx="4" fill="{TYPE_COLORS[typ]}"/>')
            if w > 28:
                parts.append(f'<text class="small" x="{x+w/2:.1f}" y="{y+18}" text-anchor="middle" fill="#ffffff">{value}</text>')
            x += w
    lx, ly = width - right + 40, top
    for idx, typ in enumerate(TYPE_ORDER):
        y = ly + idx * 24
        parts.append(f'<rect x="{lx}" y="{y-12}" width="14" height="14" rx="3" fill="{TYPE_COLORS[typ]}"/>')
        parts.append(f'<text class="axis" x="{lx+22}" y="{y}">{esc(TYPE_LABELS[typ])}</text>')
    write_svg(out, parts)


def draw_yearly_trend(records: list[dict[str, Any]], out: Path) -> None:
    years = sorted({r["adopter_year"] for r in records})
    directions = [d for d, _ in Counter(r["direction"] for r in records).most_common(4)]
    counts = {(year, d): 0 for year in years for d in directions}
    for r in records:
        if r["direction"] in directions:
            counts[(r["adopter_year"], r["direction"])] += 1
    max_count = max(counts.values()) if counts else 1
    width, height = 900, 520
    left, right, top, bottom = 80, 180, 90, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = svg_header(width, height)
    parts += [
        '<text class="title" x="40" y="42">Propagation Events over Time</text>',
        '<text class="subtitle" x="40" y="66">Year is based on adopter merge time. The four largest directions are shown.</text>',
    ]
    for t in range(0, max_count + 1, max(1, math.ceil(max_count / 5))):
        y = top + plot_h - (t / max_count) * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{t}</text>')
    def x_for(year: int) -> float:
        if len(years) == 1:
            return left + plot_w / 2
        return left + (years.index(year) / (len(years) - 1)) * plot_w
    for d in directions:
        color = DIRECTION_COLORS.get(d, "#607D8B")
        points = []
        for year in years:
            x = x_for(year)
            y = top + plot_h - (counts[(year, d)] / max_count) * plot_h
            points.append((x, y))
        path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    for year in years:
        x = x_for(year)
        parts.append(f'<text class="axis" x="{x:.1f}" y="{height-45}" text-anchor="middle">{year}</text>')
    lx, ly = width - right + 35, top
    for idx, d in enumerate(directions):
        y = ly + idx * 26
        color = DIRECTION_COLORS.get(d, "#607D8B")
        parts.append(f'<line x1="{lx}" x2="{lx+18}" y1="{y-4}" y2="{y-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="axis" x="{lx+26}" y="{y}">{esc(d)}</text>')
    parts.append(f'<text class="axis" x="{left+plot_w/2}" y="{height-16}" text-anchor="middle">Adopter merge year</text>')
    write_svg(out, parts)


def draw_file_scale(records: list[dict[str, Any]], out: Path) -> None:
    width, height = 760, 620
    left, right, top, bottom = 90, 80, 90, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    max_count = max(max(r["source_file_count"], r["target_file_count"]) for r in records)
    max_axis = max(5, min(80, max_count))
    parts = svg_header(width, height)
    parts += [
        '<text class="title" x="40" y="42">Patch Size of Propagated Changes</text>',
        '<text class="subtitle" x="40" y="66">Each point compares changed-file counts in propagator and adopter changes.</text>',
    ]
    for tick in [1, 5, 10, 20, 40, 80]:
        if tick > max_axis:
            continue
        x = left + (math.log10(tick) / math.log10(max_axis)) * plot_w
        y = top + plot_h - (math.log10(tick) / math.log10(max_axis)) * plot_h
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top+plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{x:.1f}" y="{height-52}" text-anchor="middle">{tick}</text>')
        parts.append(f'<text class="axis" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#9fb3c8" stroke-dasharray="5 5"/>')
    for r in records:
        sx = min(max_axis, max(1, r["source_file_count"]))
        ty = min(max_axis, max(1, r["target_file_count"]))
        x = left + (math.log10(sx) / math.log10(max_axis)) * plot_w
        y = top + plot_h - (math.log10(ty) / math.log10(max_axis)) * plot_h
        color = TYPE_COLORS.get(r["change_type"], "#607D8B")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" opacity="0.62"/>')
    parts.append(f'<text class="axis" x="{left+plot_w/2}" y="{height-18}" text-anchor="middle">Propagator changed-file count (log scale)</text>')
    parts.append(f'<text class="axis" transform="translate(24 {height/2}) rotate(-90)" text-anchor="middle">Adopter changed-file count (log scale)</text>')
    write_svg(out, parts)


def build_records(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, pair in enumerate(pairs, start=1):
        ordered = earlier_later(pair)
        if not ordered:
            continue
        source, target = ordered
        s_dt = side_date(source)
        t_dt = side_date(target)
        if not s_dt or not t_dt:
            continue
        delay_days = max(0.0, (t_dt - s_dt).total_seconds() / 86400)
        s_repo = repo_label(source.get("project") or "")
        t_repo = repo_label(target.get("project") or "")
        if s_repo == t_repo:
            continue
        curation = pair.get("_curation") or {}
        records.append(
            {
                "dataset_index": pair.get("_dataset_index", idx),
                "source_repo": s_repo,
                "target_repo": t_repo,
                "direction": f"{s_repo}->{t_repo}",
                "source_id": change_id(source),
                "target_id": change_id(target),
                "source_title": side_title(source),
                "target_title": side_title(target),
                "source_merged_at": s_dt.isoformat(),
                "target_merged_at": t_dt.isoformat(),
                "adopter_year": t_dt.year,
                "delay_days": delay_days,
                "delay_hours": delay_days * 24,
                "change_type": classify_change(source, target),
                "source_file_count": len(side_files(source)),
                "target_file_count": len(side_files(target)),
                "exact_file_overlap": curation.get("exact_file_overlap"),
                "file_similarity": curation.get("file_similarity"),
                "text_similarity": curation.get("text_similarity"),
                "direct_source_url_in_infestor_text": bool(curation.get("direct_source_url_in_infestor_text")),
            }
        )
    return records


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset_index",
        "source_repo",
        "target_repo",
        "direction",
        "source_id",
        "target_id",
        "source_merged_at",
        "target_merged_at",
        "delay_days",
        "change_type",
        "source_file_count",
        "target_file_count",
        "exact_file_overlap",
        "file_similarity",
        "text_similarity",
        "direct_source_url_in_infestor_text",
        "source_title",
        "target_title",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def write_report(records: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    direction_counts = Counter(r["direction"] for r in records)
    type_counts = Counter(r["change_type"] for r in records)
    delays = [r["delay_days"] for r in records]
    nonzero_delays = [d for d in delays if d > 0]
    lines = [
        "# High-Precision Propagation Empirical Analysis",
        "",
        f"Analyzed pairs: **{len(records)}**.",
        "",
        "## 1. Propagation Direction",
        "",
        "The propagation network is asymmetric. The largest directions are:",
        "",
        "| direction | count | share |",
        "|---|---:|---:|",
    ]
    for d, c in direction_counts.most_common():
        lines.append(f"| {d} | {c} | {c / len(records):.1%} |")
    lines += [
        "",
        "Figure: `fig1_direction_matrix.svg`.",
        "",
        "## 2. Propagation Time",
        "",
        f"The dataset contains **{len(delays) - len(nonzero_delays)}** zero-lag pairs, mostly imported commits whose commit timestamp is inherited from upstream. "
        "These pairs are useful evidence of code propagation, but they are not reliable for estimating integration latency.",
        "",
        f"Among **{len(nonzero_delays)}** non-zero, observable-latency pairs, median latency is **{fmt_days(median(nonzero_delays))}**. "
        f"P75 is **{fmt_days(quantile(nonzero_delays, 0.75))}**, and P90 is **{fmt_days(quantile(nonzero_delays, 0.90))}**.",
        "",
        "Figures: `fig2_latency_distribution.svg`, `fig3_latency_by_direction.svg`.",
        "",
        "## 3. Change Types",
        "",
        "Change type is assigned by title/body keyword rules, intended as a coarse empirical label.",
        "",
        "| type | count | share |",
        "|---|---:|---:|",
    ]
    for typ in TYPE_ORDER:
        c = type_counts.get(typ, 0)
        if c:
            lines.append(f"| {TYPE_LABELS[typ]} | {c} | {c / len(records):.1%} |")
    lines += [
        "",
        "Figure: `fig4_type_by_direction.svg`.",
        "",
        "## 4. Temporal Trend",
        "",
        "The yearly trend uses adopter merge time. It helps distinguish isolated backports from periods of intensive upstream synchronization.",
        "",
        "Figure: `fig5_yearly_trend.svg`.",
        "",
        "## 5. Patch Scale",
        "",
        f"Median propagator file count is **{summary['file_count']['source_median']}**, "
        f"and median adopter file count is **{summary['file_count']['target_median']}**. "
        "Large multi-file changes should be treated carefully in statement-level detection experiments.",
        "",
        "Figure: `fig6_file_scale.svg`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/URL_Results_empirical_full.json")
    parser.add_argument("--output-dir", default="outputs/empirical_analysis")
    args = parser.parse_args()

    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = build_records(pairs)
    delays = [r["delay_days"] for r in records]
    nonzero_delays = [d for d in delays if d > 0]
    summary = {
        "input": args.input,
        "raw_pairs": len(pairs),
        "analyzed_pairs": len(records),
        "direction_counts": dict(Counter(r["direction"] for r in records).most_common()),
        "type_counts": {TYPE_LABELS.get(k, k): v for k, v in Counter(r["change_type"] for r in records).most_common()},
        "latency_days": {
            "min": min(delays),
            "p25": quantile(delays, 0.25),
            "median": median(delays),
            "p75": quantile(delays, 0.75),
            "p90": quantile(delays, 0.90),
            "max": max(delays),
        },
        "observable_nonzero_latency_days": {
            "count": len(nonzero_delays),
            "zero_lag_count": len(delays) - len(nonzero_delays),
            "min": min(nonzero_delays) if nonzero_delays else 0,
            "p25": quantile(nonzero_delays, 0.25) if nonzero_delays else 0,
            "median": median(nonzero_delays) if nonzero_delays else 0,
            "p75": quantile(nonzero_delays, 0.75) if nonzero_delays else 0,
            "p90": quantile(nonzero_delays, 0.90) if nonzero_delays else 0,
            "max": max(nonzero_delays) if nonzero_delays else 0,
        },
        "file_count": {
            "source_median": median([r["source_file_count"] for r in records]),
            "target_median": median([r["target_file_count"] for r in records]),
            "source_p90": quantile([r["source_file_count"] for r in records], 0.90),
            "target_p90": quantile([r["target_file_count"] for r in records], 0.90),
        },
        "direct_url_reference_count": sum(1 for r in records if r["direct_source_url_in_infestor_text"]),
    }
    (out / "empirical_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(records, out / "empirical_records.csv")
    draw_direction_matrix(records, out / "fig1_direction_matrix.svg")
    draw_latency_histogram(records, out / "fig2_latency_distribution.svg")
    draw_latency_by_direction(records, out / "fig3_latency_by_direction.svg")
    draw_type_stacked(records, out / "fig4_type_by_direction.svg")
    draw_yearly_trend(records, out / "fig5_yearly_trend.svg")
    draw_file_scale(records, out / "fig6_file_scale.svg")
    write_report(records, summary, out / "empirical_analysis_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
