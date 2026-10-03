from __future__ import annotations
from file_queries import graphqa_files
from types import SimpleNamespace
from typing import Any

def file_action_rank(state: dict[str, Any], top: int, *, use_exact: bool=True, use_find: bool=True, use_propose: bool=True) -> list[dict[str, Any]]:
    """Compose the final file-level actions for one source-file query.

    FindCounterpart is the structural/path retrieval action. ExactCounterpart
    and ProposeNewCounterpart add explicit source-path evidence when that path
    is present or absent, respectively.
    """
    source_path = state['source_changed_paths'][0]
    target_paths = set(state['target_tree'])
    ranked = graphqa_files(state, SimpleNamespace(candidate_files=top)) if use_find else []
    by_path = {item['path']: dict(item) for item in ranked}
    promoted: list[dict[str, Any]] = []
    if use_exact and source_path in target_paths:
        item = by_path.pop(source_path, {'path': source_path, 'matched_source_path': source_path})
        item.update({'score': 1.2, 'action': 'ExactCounterpart'})
        promoted.append(item)
    elif use_propose and source_path not in target_paths:
        promoted.append({'path': source_path, 'score': 1.2, 'matched_source_path': source_path, 'action': 'ProposeNewCounterpart'})
    return (promoted + list(by_path.values()))[:top]
