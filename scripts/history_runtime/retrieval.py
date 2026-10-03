from __future__ import annotations
from typing import Any
import re
from propagation_detector import path_tokens
from propagation_detector import jaccard
from propagation_detector import path_score
TOKEN_RE = re.compile('[A-Za-z_][A-Za-z0-9_]*|\\d+')

def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or '') if len(t) >= 2}

def source_patch_text(state: dict[str, Any]) -> str:
    return '\n'.join((stmt.text for stmt in state.get('source_statements', [])[:120]))

def source_terms(state: dict[str, Any]) -> set[str]:
    out = set()
    for path in state.get('source_changed_paths', []):
        out |= path_tokens(path)
    out |= tokens(source_patch_text(state))
    title = state['propagator'].title + '\n' + state['propagator'].body + '\n' + state['propagator'].message
    out |= tokens(title)
    return {t for t in out if len(t) >= 3}

def path_prefilter(state: dict[str, Any], limit: int=120) -> list[dict[str, Any]]:
    terms = source_terms(state)
    scored = {}
    for sp in state['source_changed_paths']:
        for tp in state['target_tree']:
            score = 0.68 * path_score(sp, tp) + 0.32 * jaccard(path_tokens(tp), terms)
            old = scored.get(tp)
            if old is None or score > old['score']:
                scored[tp] = {'path': tp, 'score': score, 'matched_source_path': sp}
    return sorted(scored.values(), key=lambda x: (-x['score'], x['path']))[:limit]

def action_graphqa(state: dict[str, Any], top: int) -> list[dict[str, Any]]:
    return path_prefilter(state, top)
