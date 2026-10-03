from __future__ import annotations
import urllib.request
import json
from typing import Any
from propagation_detector import GitHubClient
import re
from graph_actions import direct_intent_ast_blocks
from repository_qa import RepositoryGraph
import time

class LLMClient:
    """Client for one user-configured OpenAI-compatible LLM endpoint."""

    def __init__(self, name: str, api_key: str, base_url: str, model: str, timeout: int=90, fallback_base_urls: list[str] | None=None) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.base_urls = [self.base_url] + [url.rstrip('/') for url in fallback_base_urls or [] if url and url.rstrip('/') != self.base_url]
        self.used_base_url = self.base_url
        self.model = model
        self.timeout = timeout
        self.last_error = ''

    def complete_json(self, system: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        body = {'model': self.model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], 'temperature': 0, 'max_tokens': 4000}
        if 'deepseek' in f'{self.model} {self.base_url}'.lower():
            body['response_format'] = {'type': 'json_object'}
            body['thinking'] = {'type': 'disabled'}
        errors = []
        for base_url in self.base_urls:
            url = base_url
            if not url.endswith('/chat/completions'):
                url += '/chat/completions'
            request = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}, method='POST')
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        data = json.loads(response.read().decode('utf-8'))
                    choice = (data.get('choices') or [{}])[0]
                    message = choice.get('message') or {}
                    content = message.get('content') or ''
                    parsed = extract_json(content)
                    if parsed is not None:
                        self.last_error = ''
                        self.used_base_url = base_url
                        self.base_urls = [base_url] + [item for item in self.base_urls if item != base_url]
                        return parsed
                    errors.append(f"{base_url}: unparseable content={content[:80]!r} finish={choice.get('finish_reason')} reasoning_len={len(message.get('reasoning_content') or '')} usage={data.get('usage')}")
                except urllib.error.HTTPError as exc:
                    errors.append(f'{base_url}: HTTP {exc.code}')
                    if exc.code < 500 and exc.code != 429:
                        break
                except Exception as exc:
                    errors.append(f'{base_url}: {type(exc).__name__}: {exc}')
                time.sleep(1.5 * (attempt + 1))
        self.last_error = ' | '.join(errors[-3:])
        return None

def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub('^```(?:json)?', '', text).strip()
    text = re.sub('```$', '', text).strip()
    for candidate in (text, text[text.find('{'):text.rfind('}') + 1]):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None

def deletion_side_blocks(state: dict[str, Any], runner: RepositoryGraph, target_paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks, intents = direct_intent_ast_blocks(state, runner, target_paths, 2)
    intents = [intent for intent in intents if intent.get('kind') == 'delete']
    retained = []
    for block in blocks:
        evidence = [intent for intent in block.get('intent_evidence', []) if intent.get('kind') == 'delete']
        if not evidence:
            continue
        item = dict(block)
        item['intent_evidence'] = evidence
        item['actions'] = ['DeleteRegion']
        retained.append(item)
    return (retained, intents)

def source_context(state: dict[str, Any], intents: list[dict[str, Any]]) -> dict[str, Any]:
    source = state['propagator']
    return {'repository': source.repo, 'title': (source.title or '')[:500], 'body': (source.body or '')[:1000], 'commit_message': (source.message or '')[:500], 'changed_files': state.get('source_changed_paths', [])[:20], 'deletion_side_intents': [{'intent_id': index, 'source_path': intent.get('path'), 'source_symbol': intent.get('symbol'), 'operation': 'edit-or-delete-existing-code', 'old_statements': intent.get('texts', [])[:5]} for index, intent in enumerate(intents[:16])]}

def block_code_preview(client: GitHubClient, state: dict[str, Any], block: dict[str, Any]) -> str:
    path = str(block.get('path') or '')
    if not path:
        return ''
    content = client.file_at(state['target'].repo, state['target'].base_sha, path) or ''
    lines = content.splitlines()
    start = max(0, int(block.get('start') or 1) - 1)
    end = min(len(lines), max(start + 1, int(block.get('end') or start + 1)))
    return '\n'.join(lines[start:end])[:1800]

def valid_ids(values: Any, upper_bound: int) -> list[int]:
    result = []
    for value in values or []:
        try:
            candidate_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= candidate_id < upper_bound and candidate_id not in result:
            result.append(candidate_id)
    return result

def llm_rerank_statements(client: GitHubClient, llm: LLMClient, state: dict[str, Any], blocks: list[dict[str, Any]], intents: list[dict[str, Any]], candidates: list[dict[str, Any]], top: int, prompt_limit: int=120) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = candidates[:prompt_limit]
    candidate_payload = [{'id': index, 'path': item.get('path'), 'line': item.get('line'), 'symbol': item.get('target_symbol'), 'statement': (item.get('line_text') or '')[:140], 'matched_source_statement': (item.get('matched_source_text') or '')[:140], 'action': item.get('action')} for index, item in enumerate(pool)]
    block_payload = [{'block_id': index, 'path': block.get('path'), 'symbol': block.get('symbol'), 'start_line': block.get('start'), 'end_line': block.get('end'), 'actions': block.get('actions', []), 'target_pre_fix_code': block_code_preview(client, state, block)} for index, block in enumerate(blocks[:24])]
    shared_context = {'source_change': source_context(state, intents), 'target_repository': {'repository': state['target'].repo, 'candidate_ast_blocks': block_payload}, 'candidate_statements': candidate_payload}
    action_payload = {'task': 'Answer each named statement-level repository-graph action question.', **shared_context, 'action_questions': {'ConstructSourceIntent': 'For each source intent, what existing behavior is edited or deleted and which symbols or operations identify it?', 'LocateAffectedBlock': 'For each source intent, which target AST block IDs contain corresponding affected behavior?', 'IdentifyDeletionRegions': 'Which target AST block IDs contain functions, blocks, APIs, or references that should be removed or substantially changed?', 'ShowCode': 'After reading the target pre-fix block and statement code, which candidate statement IDs contain the affected behavior?'}, 'output_schema': {'source_intents': [{'intent_id': 0, 'interpretation': 'concise intent', 'operation': 'edit-or-delete'}], 'LocateAffectedBlock': [{'intent_id': 0, 'block_ids': [0]}], 'IdentifyDeletionRegions': [{'intent_id': 0, 'block_ids': [0]}], 'ShowCode': {'candidate_statement_ids': [0, 1], 'evidence': 'brief explanation'}}, 'constraints': ['Use only supplied intent, block, and candidate-statement IDs.', 'Use only target pre-fix code and do not invent target change metadata.', 'Only existing statements requiring editing or deletion are in scope.', 'Return strict JSON only.']}
    action_system = 'You are the statement-level repository-graph QA agent in PropQA. Answer every named action using source intents and target pre-fix AST/code evidence. Do not output chain-of-thought. Return JSON only.'
    action_result = llm.complete_json(action_system, action_payload)
    clean_action_result: dict[str, Any] = {}
    if action_result:
        clean_action_result['source_intents'] = action_result.get('source_intents', [])[:16]
        for action in ('LocateAffectedBlock', 'IdentifyDeletionRegions'):
            mappings = []
            for mapping in action_result.get(action, []):
                try:
                    intent_id = int(mapping.get('intent_id'))
                except (TypeError, ValueError):
                    continue
                if 0 <= intent_id < len(intents):
                    mappings.append({'intent_id': intent_id, 'block_ids': valid_ids(mapping.get('block_ids'), len(block_payload))})
            clean_action_result[action] = mappings
        show_code = action_result.get('ShowCode') or {}
        clean_action_result['ShowCode'] = {'candidate_statement_ids': valid_ids(show_code.get('candidate_statement_ids'), len(candidate_payload)), 'evidence': str(show_code.get('evidence') or '')[:400]}
    finalize_payload = {'task': 'Finalize the ranked target pre-fix statements affected by the source change. Only edited or deleted existing code is in scope.', **shared_context, 'action_answers': clean_action_result, 'output_schema': {'ranked_candidate_ids': [0, 1], 'relevant_candidate_ids': [0, 1], 'brief_rationale': 'one short evidence-based sentence'}, 'constraints': ['Use only integer IDs present in candidate_statements.', 'Reconcile the answers from all statement-level graph actions.', 'Prioritize old target statements that should be edited or deleted.', 'Return strict JSON only and at most 100 ranked IDs.']}
    finalize_system = 'You are the FinalizeResults agent in PropQA. Rank affected existing target statements from the completed graph-action answers. Do not output chain-of-thought. Return JSON only.'
    result = llm.complete_json(finalize_system, finalize_payload)
    ordered_ids = []
    if result:
        for key in ('ranked_candidate_ids', 'relevant_candidate_ids'):
            for value in result.get(key, []):
                try:
                    candidate_id = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= candidate_id < len(pool) and candidate_id not in ordered_ids:
                    ordered_ids.append(candidate_id)
    selected = [pool[index] for index in ordered_ids]
    selected_keys = {(item.get('path'), item.get('line')) for item in selected}
    for item in candidates:
        key = (item.get('path'), item.get('line'))
        if key not in selected_keys:
            selected.append(item)
            selected_keys.add(key)
        if len(selected) >= top:
            break
    return (selected[:top], {'provider': llm.name, 'model': llm.model, 'base_url': llm.used_base_url, 'llm_used': action_result is not None and result is not None, 'llm_error': llm.last_error, 'selected_ids': ordered_ids[:100], 'brief_rationale': (result or {}).get('brief_rationale', ''), 'action_answers': clean_action_result, 'action_qa_used': action_result is not None, 'finalize_qa_used': result is not None})
