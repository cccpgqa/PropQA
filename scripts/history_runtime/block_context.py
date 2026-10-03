from __future__ import annotations
from typing import Any

def stmt_key(item: dict[str, Any]) -> tuple[str, int] | None:
    try:
        return (str(item['path']), int(item['line']))
    except Exception:
        return None

def unique_rank(items: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in items:
        key = stmt_key(item)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= top:
            break
    return out
