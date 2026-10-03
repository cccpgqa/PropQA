from __future__ import annotations
from typing import Any
from repository_qa import RepositoryGraph
from statement_qa import LLMClient

def compact_source_statements(state: dict[str, Any], source_file: str) -> list[dict[str, Any]]:
    return [{'operation': statement.kind, 'text': statement.text[:160]} for statement in state.get('source_statements', []) if statement.path == source_file][:16]

def candidate_symbols(runner: RepositoryGraph, state: dict[str, Any], path: str) -> list[str]:
    if path not in set(state['target_tree']):
        return []
    try:
        symbols = runner._symbols_for_file(state, 'target', state['target'].repo, state['target'].base_sha, path)
        return [str(symbol.get('name') or '') for symbol in symbols if symbol.get('name')][:8]
    except Exception:
        return []

def candidate_code_preview(runner: RepositoryGraph, state: dict[str, Any], path: str, symbols: list[str]) -> str:
    if path not in set(state['target_tree']):
        return ''
    content = runner.client.file_at(state['target'].repo, state['target'].base_sha, path) or ''
    lines = content.splitlines()
    if not lines:
        return ''
    center = 0
    for symbol in symbols:
        for index, line in enumerate(lines):
            if symbol and symbol in line:
                center = index
                break
        if center:
            break
    start = max(0, center - 4)
    return '\n'.join(lines[start:start + 16])[:1200]

def _valid_ids(values: Any, upper_bound: int) -> list[int]:
    result = []
    for value in values or []:
        try:
            candidate_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= candidate_id < upper_bound and candidate_id not in result:
            result.append(candidate_id)
    return result

def rerank_batch(llm: LLMClient, state: dict[str, Any], queries: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    source = state['propagator']
    prompt_queries = []
    candidate_lookup: dict[tuple[str, int], str] = {}
    for query_index, query in enumerate(queries):
        candidates = []
        for candidate_id, item in enumerate(query['candidates']):
            candidate_lookup[query['source_file'], candidate_id] = item['path']
            candidates.append({'id': candidate_id, 'path': item['path'], 'action': item.get('action') or 'FindCounterpart', 'structural_score': round(float(item.get('score', 0.0)), 6), 'matched_source_path': item.get('matched_source_path'), 'target_symbols': item.get('target_symbols', []), 'target_pre_fix_code': item.get('target_code_preview', '')})
        prompt_queries.append({'query_id': query_index, 'source_file': query['source_file'], 'source_changed_statements': query['source_statements'], 'candidate_target_files': candidates})
    shared_context = {'source_change': {'repository': source.repo, 'title': (source.title or '')[:500], 'body': (source.body or '')[:1000], 'commit_message': (source.message or '')[:500]}, 'target_repository': state['target'].repo, 'file_queries': prompt_queries}
    action_payload = {'task': 'Answer each named repository-graph action question for every source-file query.', **shared_context, 'action_questions': {'AnalyzeSourceChange': 'What behavior, symbols, and file role are changed by this source file?', 'LocateFile': 'Which candidate IDs match the source path, basename, extension, or directory role?', 'LocateCounterpart': 'Which existing candidate IDs implement the same responsibility as the source file?', 'InferCounterpart': 'If no existing counterpart is adequate, what source-derived target path should be considered?', 'LocateSymbol': 'Which candidate IDs contain target symbols corresponding to changed source symbols?', 'LocateCodeSnippet': 'Which candidate code previews contain source-like structures or operations?', 'SemanticSearch': 'Which candidates implement the source change intent despite lexical or path differences?', 'ShowCode': 'After inspecting the supplied target pre-fix previews, which candidates retain affected behavior?', 'LocateRelatedArtifacts': 'Which candidate IDs are related tests, fixtures, generated outputs, or data companions?'}, 'output_schema': {'action_answers': [{'query_id': 0, 'source_intent': 'concise answer to AnalyzeSourceChange', 'LocateFile': [0], 'LocateCounterpart': [0], 'InferCounterpart': 'path or empty string', 'LocateSymbol': [0], 'LocateCodeSnippet': [0], 'SemanticSearch': [0], 'ShowCode': [0], 'LocateRelatedArtifacts': [], 'evidence': 'brief grounded explanation'}]}, 'constraints': ['Return one action answer for every query_id and every named action.', 'For ID-valued answers, use only candidate IDs listed under that query.', 'Use only source metadata and target pre-fix graph evidence supplied here.', 'Return strict JSON only.']}
    action_system = 'You are a repository-graph QA agent for cross-repository patch propagation. Answer each graph action question using file-tree, AST-symbol, and code-preview evidence. Do not output chain-of-thought. Return JSON only.'
    action_response = llm.complete_json(action_system, action_payload)
    answers_by_query: dict[int, dict[str, Any]] = {}
    if action_response:
        for item in action_response.get('action_answers', []):
            try:
                query_id = int(item.get('query_id'))
            except (TypeError, ValueError):
                continue
            if not 0 <= query_id < len(queries):
                continue
            limit = len(queries[query_id]['candidates'])
            clean = dict(item)
            for action in ('LocateFile', 'LocateCounterpart', 'LocateSymbol', 'LocateCodeSnippet', 'SemanticSearch', 'ShowCode', 'LocateRelatedArtifacts'):
                clean[action] = _valid_ids(item.get(action), limit)
            answers_by_query[query_id] = clean
    finalize_payload = {'task': 'Finalize the affected target-file ranking from the completed graph-action answers.', **shared_context, 'action_answers': [{'query_id': query_id, **answers_by_query.get(query_id, {})} for query_id in range(len(queries))], 'output_schema': {'rankings': [{'query_id': 0, 'ranked_candidate_ids': [0, 1, 2], 'brief_rationale': 'one short evidence-based sentence'}]}, 'constraints': ['Return one ranking for every query_id.', 'Use only candidate IDs listed under that query.', 'Reconcile evidence across all action answers rather than relying on path alone.', 'Return strict JSON only.']}
    finalize_system = 'You are the FinalizeResults agent in PropQA. Use the answers produced by each graph action to rank affected target files. Do not output chain-of-thought. Return JSON only.'
    final_response = llm.complete_json(finalize_system, finalize_payload)
    by_query: dict[str, list[int]] = {}
    rationales: dict[str, str] = {}
    if final_response:
        for item in final_response.get('rankings', []):
            try:
                query_id = int(item.get('query_id'))
            except (TypeError, ValueError):
                continue
            if not 0 <= query_id < len(queries):
                continue
            source_file = queries[query_id]['source_file']
            by_query[source_file] = _valid_ids(item.get('ranked_candidate_ids'), len(queries[query_id]['candidates']))
            rationales[source_file] = str(item.get('brief_rationale') or '')[:300]
    predictions = {}
    for query in queries:
        source_file = query['source_file']
        ranked_ids = by_query.get(source_file, [])
        ranked_paths = [candidate_lookup[source_file, candidate_id] for candidate_id in ranked_ids]
        for item in query['candidates']:
            if item['path'] not in ranked_paths:
                ranked_paths.append(item['path'])
        predictions[source_file] = ranked_paths[:5]
    return (predictions, {'provider': llm.name, 'model': llm.model, 'base_url': llm.used_base_url, 'llm_used': action_response is not None and final_response is not None, 'llm_error': llm.last_error, 'returned_queries': len(by_query), 'expected_queries': len(queries), 'rationales': rationales, 'action_answers': answers_by_query, 'action_qa_used': action_response is not None, 'finalize_qa_used': final_response is not None})
