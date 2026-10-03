"""AST-owned statement context and bounded QA payloads."""
import json
import math
QUESTION = 'Does this source change require edits or deletions to existing target statements? Analyze the target blocks as a whole using the source patch, source intent, AST node ownership and containment edges. Then select affected statement IDs. Source additions may require target edits; do not report locations for purely new target statements. A control node owns its header, not child statements: child edits alone do not affect its parent. Ignore comment/format-only changes and pure moves. Consider every listed node. Return JSON {"complete":true,"reviewed_count":N, "affected":[[id,confidence,"edit" or "delete"],...],"evidence":"brief grounded justification"}. List possible affected nodes with confidence in (0,1]. Unlisted nodes are explicitly judged unaffected (score zero), not skipped. Empty affected is valid. No positive quota. Use only supplied IDs. Do not repeat code or provide a reasoning transcript.'

def validate_answer(answer, count):
    if not isinstance(answer, dict) or answer.get('complete') is not True:
        raise ValueError('Incomplete block review')
    if type(answer.get('reviewed_count')) is not int or answer['reviewed_count'] != count:
        raise ValueError('Incorrect reviewed node count')
    rows = answer.get('affected')
    if not isinstance(rows, list):
        raise ValueError('Missing affected array')
    seen = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError('Invalid affected row')
        i, score, operation = row
        if type(i) is not int or not 0 <= i < count or i in seen:
            raise ValueError('Unknown or duplicate statement ID')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 < score <= 1:
            raise ValueError('Invalid confidence')
        if operation not in ('edit', 'delete'):
            raise ValueError('Target insertions are outside scope')
        seen.add(i)
    if not isinstance(answer.get('evidence'), str) or len(answer['evidence']) > 4000:
        raise ValueError('Missing or excessive evidence')

def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]+1:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged

def context_for(nodes, blocks, contents):
    """Code is emitted once per merged region; AST nodes refer to line ownership."""
    files, edges, mapping = [], [], []
    for i, u in enumerate(nodes):
        mapping.append({'id': i, 'path': u['path'], 'kind': u['kind'],
                        'start': u['start_line'], 'end': u['end_line'],
                        'owned_lines': u['owned_lines']})
        parents = [(j, v) for j,v in enumerate(nodes) if j != i and v['path'] == u['path']
                   and v['start_byte'] <= u['start_byte'] and v['end_byte'] >= u['end_byte']
                   and (v['start_byte'], v['end_byte']) != (u['start_byte'], u['end_byte'])]
        if parents:
            j, _ = min(parents, key=lambda item:item[1]['end_byte']-item[1]['start_byte'])
            edges.append([j, i])
    for path in sorted({u['path'] for u in nodes}):
        lines = contents[path].splitlines()
        local = [u for u in nodes if u['path'] == path]
        # Send complete blocks where modest; oversized blocks use explicit code windows.
        regions = []
        for u in local:
            matches = [b for b in blocks if b['path'] == path and b['start'] <= u['start_line'] <= b['end']]
            b = min(matches, key=lambda x:x['end']-x['start']) if matches else None
            if b and b['end']-b['start'] < 240:
                regions.append([b['start'], b['end']])
            else:
                regions.extend([max(1, line-4), min(len(lines), line+4)] for line in u['owned_lines'])
        files.append({'path': path, 'code_regions': [
            {'start_line': a, 'end_line': b, 'code': '\n'.join(f'{i+1}: {lines[i]}' for i in range(a-1,b))}
            for a,b in merge_ranges(regions)],
            'blocks': [{k:b[k] for k in ('symbol','start','end')} for b in blocks
                       if b['path'] == path and any(b['start'] <= u['start_line'] <= b['end'] for u in local)]})
    return {'files': files, 'nodes': mapping, 'ast_contains_edges': edges}

def payload_for(source, nodes, blocks, contents):
    return {'source': source, 'target': context_for(nodes, blocks, contents),
            'question': QUESTION, 'required_node_count': len(nodes)}

def pack(source, candidates, blocks, contents):
    batches, pending = [], [candidates]
    while pending:
        nodes = pending.pop(0)
        if not nodes:
            continue
        payload = payload_for(source, nodes, blocks, contents)
        size = len(json.dumps(payload).encode())
        if len(nodes) <= 160 and size <= 85000:
            batches.append({'ids': [u['id'] for u in nodes], 'payload': payload,
                            'payload_bytes': size})
        elif len(nodes) == 1:
            raise ValueError('Source/block context exceeds budget; no silent truncation')
        else:
            mid = len(nodes)//2
            pending[0:0] = [nodes[:mid], nodes[mid:]]
    return batches
