from __future__ import annotations
from block_context import unique_rank
from typing import Any

def evidence_gated(preds: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    """Remove broad-block noise while retaining strong semantic/seed evidence."""
    kept = []
    for item in preds:
        sim = float(item.get('line_similarity', 0.0) or 0.0)
        prox = float(item.get('proximity_score', 0.0) or 0.0)
        modules = len(item.get('statement_modules', []))
        block_score = float(item.get('block_score', 0.0) or 0.0)
        if sim >= 0.18 or prox >= 0.75 or (sim >= 0.1 and prox >= 0.45) or (modules >= 3 and block_score >= 0.9 and (prox >= 0.25)):
            kept.append(item)
    return unique_rank(kept, top)
