"""Expand a frozen graph-selected seed using source AST statements."""
import math
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from tree_sitter import Language, Parser
import tree_sitter_go
from ast_statement_protocol import extract
from snapshots import git
from statement_retrieval import terms

def parse(path, text):
    ast = extract(path, text)
    raw = text.encode()
    root = Parser(Language(tree_sitter_go.language())).parse(raw).root_node
    functions, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.type in ('function_declaration', 'method_declaration'):
            name = node.child_by_field_name('name')
            functions.append((node.start_byte, node.end_byte, raw[name.start_byte:name.end_byte].decode()))
        stack.extend(node.named_children)
    result = {}
    for u in ast['units']:
        own = ' '.join(ast['tokens'][i]['value'] for i in u['token_indices'])
        enclosing = [f for f in functions if f[0] <= u['start_byte'] and f[1] >= u['end_byte']]
        symbol = min(enclosing, key=lambda f: f[1]-f[0])[2] if enclosing else ''
        result[u['id']] = dict(u, own=own, terms=terms(own), symbol=symbol)
    return result

def select(inference, original):
    nodes, source = inference['target_statements'], inference['source']
    targets = {}
    for path in sorted({u['path'] for u in nodes}):
        targets.update(parse(path, git(inference['target_repo'], 'show', f'{inference["target_base"]}:{path}')))
    targets = {u['id']: targets[u['id']] for u in nodes}
    source_nodes = {}
    for path in sorted({u['path'] for u in source['changed_statements']}):
        source_nodes.update(parse(path, git(source['repo'], 'show', f'{source["base_sha"]}:{path}')))
    queries = { (source_nodes[u['id']]['path'], source_nodes[u['id']]['symbol'], source_nodes[u['id']]['own']):
                source_nodes[u['id']] for u in source['changed_statements'] }
    df = Counter(t for u in targets.values() for t in u['terms'])
    idf = {t: math.log((1+len(targets))/(1+n))+1 for t, n in df.items()}
    inverted = defaultdict(list)
    for ident, u in targets.items():
        v = {t: (1+math.log(n))*idf[t] for t, n in u['terms'].items()}
        norm = math.sqrt(sum(x*x for x in v.values())) or 1
        for t, x in v.items():
            inverted[t].append((ident, x/norm))
    blocks = {}
    for batch in original['batches']:
        for file in batch['payload']['target']['files']:
            for b in file['blocks']:
                key = (file['path'], b['symbol'], b['start'], b['end'])
                blocks[key] = [i for i, u in targets.items() if u['path'] == file['path'] and
                               any(b['start'] <= line <= b['end'] for line in u['owned_lines'])]
    cosine, contextual, queues, admissible = defaultdict(float), defaultdict(float), [], defaultdict(float)
    for q in queries.values():
        qv = {t: (1+math.log(n))*idf.get(t, math.log(1+len(targets))+1) for t,n in q['terms'].items()}
        norm = math.sqrt(sum(x*x for x in qv.values())) or 1
        scores = defaultdict(float)
        for t,x in qv.items():
            for ident,y in inverted.get(t, []):
                scores[ident] += x*y/norm
        for ident, value in scores.items():
            u = targets[ident]
            same_symbol = bool(q['symbol']) and q['symbol'] == u['symbol']
            same_path = PurePosixPath(q['path']).name == PurePosixPath(u['path']).name
            rank = .80*value + .12*same_symbol + .05*same_path + .03*(q['kind']==u['kind'])
            cosine[ident] = max(cosine[ident], value)
            contextual[ident] = max(contextual[ident], rank)
        block_order = sorted(blocks, key=lambda b: (-max((scores[i] for i in blocks[b]), default=0), b))[:2]
        local = {i for b in block_order for i in blocks[b] if scores[i] > 0}
        ranked = sorted(local, key=lambda i: (-scores[i], -int(targets[i]['symbol']==q['symbol']), i))
        if ranked:
            cutoff = max(.60, .85*scores[ranked[0]])
            accepted = [i for i in ranked if scores[i] >= cutoff and
                        (len(q['terms']) >= 4 or (bool(q['symbol']) and targets[i]['symbol']==q['symbol']))][:4]
            queues.append(accepted)
            for i in accepted:
                admissible[i] = max(admissible[i], contextual[i])
    ordered = lambda v: sorted(v, key=lambda i: (-v[i], i))
    base = set(original['candidate_ids'])
    extra = [i for i in ordered(admissible) if i not in base]
    chosen = sorted(base | set(extra[:min(240, max(12, 4*len(queries)))]))
    assert len(chosen) <= 360
    return chosen
