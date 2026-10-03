from __future__ import annotations
from statement_context import evidence_gated
from typing import Any
from collections import defaultdict
from repository_qa import meaningful_statement
from repository_qa import classify_patch_pattern
from propagation_detector import GitHubClient
from propagation_detector import token_similarity
from repository_qa import pattern_symbol_score
from block_context import unique_rank
from propagation_detector import path_score
from repository_qa import RepositoryGraph
from propagation_detector import path_tokens

def source_intents(state: dict[str, Any], runner: RepositoryGraph) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for stmt in state.get('source_statements', []):
        if stmt.kind not in {'added', 'deleted'} or not meaningful_statement(stmt.text):
            continue
        source_sha = state['propagator'].merged_sha if stmt.kind == 'added' else state['propagator'].base_sha
        symbols = runner._symbols_for_file(state, 'source', state['propagator'].repo, source_sha, stmt.path)
        symbol = next((sym for sym in symbols if int(sym['start_line']) <= stmt.line_no <= int(sym['end_line'])), None)
        symbol_name = (symbol or {}).get('name') or f'line_{stmt.line_no // 20}'
        key = (stmt.path, symbol_name, stmt.kind)
        item = grouped.setdefault(key, {'key': '::'.join(key), 'path': stmt.path, 'symbol': symbol_name, 'kind': 'insert' if stmt.kind == 'added' else 'delete', 'texts': [], 'lines': [], 'source_symbol': symbol})
        item['texts'].append(stmt.text[:500])
        item['lines'].append(stmt.line_no)
    return list(grouped.values())

def symbol_match_score(intent: dict[str, Any], target_symbol: dict[str, Any], source_path: str, target_path: str) -> float:
    source_symbol = intent.get('source_symbol') or {}
    source_text = '\n'.join(intent['texts'][:12])
    target_text = target_symbol.get('text', '')
    text_sim = token_similarity(source_text, target_text)
    name_sim = token_similarity(str(source_symbol.get('name') or intent['symbol']), str(target_symbol.get('name') or ''))
    source_tokens = set(source_symbol.get('tokens') or path_tokens(source_path))
    target_tokens = set(target_symbol.get('tokens') or path_tokens(target_path))
    token_overlap = len(source_tokens & target_tokens) / max(len(source_tokens | target_tokens), 1)
    pattern = classify_patch_pattern([type('IntentStmt', (), {'text': text})() for text in intent['texts'][:20]])
    pattern_score = pattern_symbol_score(pattern, target_symbol)
    return 0.3 * text_sim + 0.27 * name_sim + 0.18 * token_overlap + 0.15 * path_score(source_path, target_path) + 0.1 * pattern_score

def direct_intent_ast_blocks(state: dict[str, Any], runner: RepositoryGraph, target_paths: list[str], per_intent_file_quota: int, use_delete_region: bool=True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate blocks directly from source intents, without global line candidates."""
    intents = source_intents(state, runner)
    blocks_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for intent in intents:
        for target_path in target_paths:
            symbols = runner._symbols_for_file(state, 'target', state['target'].repo, state['target'].base_sha, target_path)
            if not symbols:
                content = runner.client.file_at(state['target'].repo, state['target'].base_sha, target_path) or ''
                symbols = [{'name': 'file', 'start_line': 1, 'end_line': max(len(content.splitlines()), 1), 'text': content[:4000]}]
            scored = [(symbol_match_score(intent, sym, intent['path'], target_path), sym) for sym in symbols]
            scored.sort(key=lambda item: (-item[0], int(item[1]['start_line'])))
            for score, sym in scored[:per_intent_file_quota]:
                key = (target_path, int(sym['start_line']), int(sym['end_line']))
                block = blocks_by_key.setdefault(key, {'path': target_path, 'symbol': sym.get('name') or 'file', 'start': int(sym['start_line']), 'end': int(sym['end_line']), 'score': 0.0, 'support_modules': ['stmt_graphqa_direct_ast'], 'candidate_lines': [], 'items': [], 'intent_evidence': [], 'actions': []})
                action = 'DeleteRegion' if use_delete_region and intent['kind'] == 'delete' else 'FindAffectedASTBlock'
                block['score'] = max(float(block['score']), score + (0.12 if action == 'DeleteRegion' else 0.0))
                block['actions'].append(action)
                block['intent_evidence'].append(intent)
    blocks = list(blocks_by_key.values())
    for block in blocks:
        block['actions'] = sorted(set(block['actions']))
        block['intent_evidence'].sort(key=lambda intent: intent['key'])
    blocks.sort(key=lambda block: (-float(block['score']), block['path'], int(block['start'])))
    return (blocks, intents)

def expand_direct_blocks(client: GitHubClient, state: dict[str, Any], blocks: list[dict[str, Any]], intents: list[dict[str, Any]], dynamic_top: int, per_intent_cap: int, use_delete_region: bool=True, use_insert_anchor: bool=True, use_evidence_gate: bool=False) -> list[dict[str, Any]]:
    ranked = []
    for block_rank, block in enumerate(blocks, 1):
        content = client.file_at(state['target'].repo, state['target'].base_sha, block['path']) or ''
        lines = content.splitlines()
        if not lines:
            continue
        block_intents = block.get('intent_evidence') or intents
        for intent in block_intents:
            texts = intent['texts'][:16]
            action = 'DeleteRegion' if use_delete_region and intent['kind'] == 'delete' else 'FindAffectedStatement'
            for line_no in range(max(1, block['start']), min(len(lines), block['end']) + 1):
                text = lines[line_no - 1].strip()
                if not meaningful_statement(text):
                    continue
                sim = max((token_similarity(src, text) for src in texts), default=0.0)
                score = 0.42 * float(block['score']) + 0.48 * sim + 0.1 / (block_rank + 1)
                ranked.append({'path': block['path'], 'line': line_no, 'score': round(score, 6), 'line_similarity': round(sim, 6), 'proximity_score': 0.75 if action == 'DeleteRegion' else 0.45, 'line_text': text[:240], 'target_symbol': block['symbol'], 'block_start': block['start'], 'block_end': block['end'], 'block_score': block['score'], 'statement_modules': ['stmt_graphqa_direct_ast'], 'matched_source_text': texts[0][:240] if texts else '', 'intent_key': intent['key'], 'intent_budget': min(120, max(per_intent_cap, len(texts) * (3 if action == 'DeleteRegion' else 1))), 'action': action})
            if use_insert_anchor and intent['kind'] == 'insert':

                def clamp_line(value: int) -> int:
                    return min(len(lines), max(1, int(value)))
                anchors = {clamp_line(block['start']), clamp_line(block['end'])}
                anchors.update({clamp_line(block['start'] + 1), clamp_line(block['end'] - 1)})
                for line_no in sorted(anchors):
                    ranked.append({'path': block['path'], 'line': line_no, 'score': round(0.52 * float(block['score']) + 0.18, 6), 'line_similarity': 0.2, 'proximity_score': 1.0, 'line_text': lines[line_no - 1].strip()[:240], 'target_symbol': block['symbol'], 'block_start': block['start'], 'block_end': block['end'], 'block_score': block['score'], 'statement_modules': ['stmt_graphqa_direct_ast', 'InsertAnchor'], 'matched_source_text': texts[0][:240] if texts else '', 'intent_key': intent['key'], 'intent_budget': max(per_intent_cap, len(texts)), 'action': 'InsertAnchor'})
    ranked.sort(key=lambda item: (-float(item['score']), item['path'], int(item['line'])))
    candidates = unique_rank(ranked, dynamic_top * 3)
    gated = evidence_gated(candidates, dynamic_top * 3) if use_evidence_gate else candidates
    return round_robin_dynamic_intent(gated, dynamic_top, per_intent_cap)

def round_robin_dynamic_intent(preds: list[dict[str, Any]], top: int, default_cap: int) -> list[dict[str, Any]]:
    """Round-robin with larger quotas for explicit DeleteRegion intents."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    caps: dict[str, int] = {}
    for item in preds:
        key = str(item.get('intent_key') or f"{item['path']}:{item.get('target_symbol')}")
        groups[key].append(item)
        caps[key] = max(caps.get(key, default_cap), int(item.get('intent_budget', default_cap)))
    for key in groups:
        groups[key].sort(key=lambda item: (-float(item['score']), item['path'], int(item['line'])))
        groups[key] = groups[key][:caps[key]]
    keys = sorted(groups, key=lambda key: -float(groups[key][0]['score']) if groups[key] else 0.0)
    out = []
    while len(out) < top and keys:
        progressed = False
        for key in list(keys):
            if groups[key]:
                out.append(groups[key].pop(0))
                progressed = True
                if len(out) >= top:
                    break
            if not groups[key]:
                keys.remove(key)
        if not progressed:
            break
    return unique_rank(out, top)
