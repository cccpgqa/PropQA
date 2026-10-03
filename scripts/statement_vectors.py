"""Statement-subtree code2vec transfer; no new supervised fitting."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from go_code2vec import (GoCode2VecEncoder, parser, walk, LEAF_TYPES, ast_path,
                        terminal_token, vectorize_contexts)

ATOMIC = {'expression_statement', 'assignment_statement', 'short_var_declaration',
          'return_statement', 'inc_statement', 'dec_statement', 'send_statement',
          'defer_statement', 'go_statement', 'break_statement', 'continue_statement',
          'var_spec', 'const_spec', 'type_spec'}




def statements(text):
    raw = text.encode('utf-8')
    tree = parser().parse(raw)
    if tree.root_node.has_error:
        raise ValueError('Go parse error in statement extraction')
    nodes = {}
    for node in walk(tree.root_node):
        if node.type in ATOMIC:
            nodes[(node.start_byte, node.end_byte)] = node
        if node.type in {'if_statement', 'for_statement', 'expression_switch_statement'}:
            condition = node.child_by_field_name('condition') or node.child_by_field_name('value')
            if condition is not None:
                nodes[(condition.start_byte, condition.end_byte)] = condition
    return raw, [nodes[k] for k in sorted(nodes)]

def encode_statements(encoder, text, canonical_units=None):
    raw, nodes = statements(text)
    if canonical_units is not None:
        wanted = {(u['start_byte'], u['end_byte']): u['id'] for u in canonical_units}
        tree = parser().parse(raw)
        nodes = [n for n in walk(tree.root_node) if (n.start_byte, n.end_byte) in wanted]
    examples = []
    for node in nodes:
        # Avoid encoding nested bodies as a single huge statement.
        leaves = []
        def visit(n):
            if n.id != node.id and n.type in {'block', 'func_literal'}:
                return
            if canonical_units is not None and n.id != node.id and (n.type.endswith('_statement') or n.type == 'short_var_declaration'):
                return
            if n.type in LEAF_TYPES:
                leaves.append(n)
                return
            for child in n.children:
                visit(child)
        visit(node)
        seed = int(hashlib.sha256(raw[node.start_byte:node.end_byte]).hexdigest()[:8], 16)
        rng = random.Random(seed)
        if len(leaves) > 60:
            leaves = [leaves[i] for i in sorted(rng.sample(range(len(leaves)), 60))]
        contexts = []
        for i, left in enumerate(leaves):
            for right in leaves[i + 1:i + 31]:
                path = ast_path(left, right, node, int(encoder.config['max_path_length']))
                if path:
                    contexts.append((terminal_token(raw, left), path, terminal_token(raw, right)))
        if not contexts:
            continue
        limit = int(encoder.config['max_contexts'])
        if len(contexts) > limit:
            contexts = rng.sample(contexts, limit)
        examples.append({'start': node.start_point.row + 1, 'end': node.end_point.row + 1,
                         'id': wanted[(node.start_byte, node.end_byte)] if canonical_units is not None else None,
                         'contexts': contexts})
    output = []
    for offset in range(0, len(examples), 128):
        batch = examples[offset:offset + 128]
        arrays = [vectorize_contexts(e['contexts'], encoder.vocab, int(encoder.config['max_contexts'])) for e in batch]
        with torch.no_grad():
            vectors = encoder.model.encode_contexts(*[
                torch.from_numpy(np.stack([a[i] for a in arrays])) for i in range(4)
            ]).numpy()
        for example, vector in zip(batch, vectors):
            norm = np.linalg.norm(vector)
            if norm:
                output.append({'start': example['start'], 'end': example['end'],
                               'id': example['id'], 'vector': vector / norm})
    return output
