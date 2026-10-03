"""Go pre-change AST statement units, token-owned labels, and exact-node mapping.

Control statements own their headers, not statements nested in their bodies.
Comments, formatting and standalone additions do not label old statements.
Function signatures and package declarations are audited outside this scope.
"""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from tree_sitter import Language, Parser
import tree_sitter_go

VERSION = 'go-ast-statement-v1'
SIMPLE = {'short_var_declaration', 'var_declaration', 'const_declaration', 'type_declaration'}
IGNORE = {'comment'}


def extract(path, text):
    raw = text.encode('utf-8')
    tree = Parser(Language(tree_sitter_go.language())).parse(raw)
    if tree.root_node.has_error:
        raise ValueError(f'Go AST parse error: {path}')
    units, tokens = [], []

    def visit(node, owner=None, in_function=False):
        if node.type in IGNORE:
            return
        in_function = in_function or node.type in {'function_declaration', 'method_declaration', 'func_literal'}
        is_stmt = in_function and (node.type.endswith('_statement') or node.type in SIMPLE)
        # Labels are wrappers; the actual child statement is the prediction unit.
        is_stmt = is_stmt and node.type not in {'empty_statement', 'labeled_statement'}
        if is_stmt:
            owner = len(units)
            units.append({'id': f'{path}@{node.start_byte}:{node.end_byte}:{node.type}',
                          'path': path, 'kind': node.type,
                          'start_byte': node.start_byte, 'end_byte': node.end_byte,
                          'start_line': node.start_point.row+1, 'end_line': node.end_point.row+1,
                          'text': raw[node.start_byte:node.end_byte].decode(), 'token_indices': []})
        if not node.children:
            value = raw[node.start_byte:node.end_byte].decode()
            if not value or (not node.is_named and value in {'{', '}', ';'}):
                return
            idx = len(tokens)
            tokens.append({'value': value, 'owner': owner, 'start_byte': node.start_byte,
                           'end_byte': node.end_byte, 'line': node.start_point.row+1,
                           'end_line': node.end_point.row+1})
            if owner is not None:
                units[owner]['token_indices'].append(idx)
            return
        for child in node.children:
            visit(child, owner, in_function)

    visit(tree.root_node)
    for unit in units:
        unit['owned_lines'] = sorted({line for i in unit['token_indices']
                                     for line in range(tokens[i]['line'], tokens[i]['end_line']+1)})
    return {'units': units, 'tokens': tokens}


def label_changes(path, before, after):
    old, new = extract(path, before), extract(path, after)
    a, b = old['tokens'], new['tokens']
    matches, changed, replacements, outside = defaultdict(Counter), set(), set(), []
    for tag, i, j, k, l in SequenceMatcher(None, [t['value'] for t in a],
                                          [t['value'] for t in b], autojunk=False).get_opcodes():
        if tag == 'equal':
            for x, y in zip(a[i:j], b[k:l]):
                if x['owner'] is not None and y['owner'] is not None:
                    matches[x['owner']][y['owner']] += 1
        else:
            owners = {t['owner'] for t in a[i:j] if t['owner'] is not None}
            changed.update(owners)
            if tag == 'replace':
                replacements.update(owners)
            if any(t['owner'] is None for t in a[i:j]):
                outside.append({'old_token_start': i, 'old_token_end': j, 'operation': tag})
    for idx, unit in enumerate(old['units']):
        candidates = matches[idx]
        counterpart = candidates.most_common(1)[0][0] if candidates else None
        signature = [a[i]['value'] for i in unit['token_indices']]
        new_signature = ([b[i]['value'] for i in new['units'][counterpart]['token_indices']]
                         if counterpart is not None else None)
        # Equal-token alignment also detects insertion of an argument within an existing statement.
        affected = bool(signature) and (idx in changed or (counterpart is not None and signature != new_signature))
        unit['label'] = int(affected)
        unit['operation'] = ('edited' if counterpart is not None or idx in replacements else 'deleted') if affected else 'unchanged'
        unit['alignment_ambiguous'] = len(candidates) > 1
    return old, {'outside_statement_changes': outside,
                 'ambiguous_alignments': sum(u['alignment_ambiguous'] for u in old['units'])}


def line_mapping(ast, line):
    """Legacy line adapter: no tolerance or label-driven choice of a node."""
    return [u['id'] for u in ast['units'] if line in u['owned_lines']]


def exact_predictions(units, predicted_ids):
    known = {u['id'] for u in units}
    predicted = set(predicted_ids)
    if predicted - known:
        raise ValueError('Predictions contain unknown AST IDs')
    return {u['id']: int(u['id'] in predicted) for u in units}
