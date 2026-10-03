from __future__ import annotations
from typing import Any
from retrieval import action_graphqa
import argparse

def restrict_state_to_source_file(state: dict[str, Any], source_file: str) -> dict[str, Any]:
    new_state = dict(state)
    new_state['source_changed_paths'] = [source_file]
    new_state['source_files'] = [item for item in state.get('source_files', []) if item.get('filename') == source_file]
    new_state['source_statements'] = [stmt for stmt in state.get('source_statements', []) if getattr(stmt, 'path', None) == source_file]
    return new_state

def graphqa_files(state: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return action_graphqa(state, args.candidate_files)
