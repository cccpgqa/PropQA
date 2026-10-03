from __future__ import annotations
from typing import Any
from propagation_detector import meaningful_statement
from pathlib import Path
import re
import os
TOKEN_RE = re.compile('[A-Za-z_][A-Za-z0-9_]*|\\d+')
GO_SYMBOL_TYPES = {'function_declaration', 'method_declaration', 'type_declaration', 'const_declaration', 'var_declaration'}

class RepositoryGraph:

    def __init__(self, client, **kwargs):
        self.client = client
        self.ast = ASTAnalyzer()

    def _symbols_for_file(self, state: dict[str, Any], role: str, repo: str, sha: str, path: str) -> list[dict[str, Any]]:
        cache_name = 'source_symbol_cache' if role == 'source' else 'target_symbol_cache'
        cache = state[cache_name]
        key = (repo, sha, path)
        if key in cache:
            return cache[key]
        content = self.client.file_at(repo, sha, path)
        symbols = self.ast.extract_symbols(path, content or '')
        cache[key] = symbols
        return symbols

class ASTAnalyzer:

    def __init__(self) -> None:
        self.enabled = False
        self.parser = None
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_go
            self.parser = Parser(Language(tree_sitter_go.language()))
            self.enabled = True
        except Exception:
            self.enabled = False

    def extract_symbols(self, path: str, content: str) -> list[dict[str, Any]]:
        if not content:
            return []
        if not self.enabled or not path.endswith('.go'):
            return fallback_go_symbols(content)
        try:
            tree = self.parser.parse(content.encode('utf-8'))
            root = tree.root_node
            symbols = []
            stack = list(root.children)
            while stack:
                node = stack.pop()
                if node.type in GO_SYMBOL_TYPES:
                    text = node.text.decode('utf-8', errors='replace')
                    symbols.append({'name': extract_symbol_name(text, node.type), 'kind': node.type, 'start_line': node.start_point[0] + 1, 'end_line': node.end_point[0] + 1, 'text': text[:4000], 'tokens': sorted(set(TOKEN_RE.findall(text.lower()))), 'ast_types': ast_type_counts(node)})
                stack.extend(node.children)
            return symbols
        except Exception:
            return fallback_go_symbols(content)

def ast_type_counts(node: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    stack = [node]
    while stack:
        current = stack.pop()
        counts[current.type] = counts.get(current.type, 0) + 1
        stack.extend(current.children)
    return counts

def fallback_go_symbols(content: str) -> list[dict[str, Any]]:
    symbols = []
    lines = content.splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not (stripped.startswith('func ') or stripped.startswith('type ') or stripped.startswith('var ') or stripped.startswith('const ')):
            continue
        end = min(len(lines), idx + 80)
        text = '\n'.join(lines[idx - 1:end])
        symbols.append({'name': extract_symbol_name(stripped, 'fallback'), 'kind': 'fallback', 'start_line': idx, 'end_line': end, 'text': text[:4000], 'tokens': sorted(set(TOKEN_RE.findall(text.lower()))), 'ast_types': {}})
    return symbols

def extract_symbol_name(text: str, kind: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ''
    if kind == 'method_declaration' or first.startswith('func ('):
        match = re.search('\\)\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\(', first)
        if match:
            return match.group(1)
    if kind == 'function_declaration' or first.startswith('func '):
        match = re.search('func\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(', first)
        if match:
            return match.group(1)
    match = re.search('(?:type|var|const)\\s+([A-Za-z_][A-Za-z0-9_]*)', first)
    if match:
        return match.group(1)
    return first[:80]

def classify_patch_pattern(source_statements: list[Any]) -> str:
    text = '\n'.join((stmt.text.lower() for stmt in source_statements[:80]))
    if any((word in text for word in ['limit', 'max', 'cap', 'bound', 'too many', 'percentile'])):
        return 'bounds_or_limit'
    if any((word in text for word in ['lock', 'unlock', 'mutex', 'atomic', 'race', 'deadlock', 'channel', 'close'])):
        return 'concurrency'
    if any((word in text for word in ['err', 'error', 'return nil', 'panic', 'recover'])):
        return 'error_handling'
    if any((word in text for word in ['cache', 'freezer', 'database', 'sync', 'flush'])):
        return 'storage_cache'
    if any((word in text for word in ['test', 'assert', 'require.', 't.'])):
        return 'test_change'
    return 'generic'

def pattern_symbol_score(pattern: str, symbol: dict[str, Any]) -> float:
    text = (symbol.get('text') or '').lower()
    return pattern_line_score(pattern, text)

def pattern_line_score(pattern: str, text: str) -> float:
    text = text.lower()
    keywords = {'bounds_or_limit': ['limit', 'max', 'cap', 'bound', 'len', 'percentile', 'range'], 'concurrency': ['lock', 'unlock', 'mutex', 'atomic', 'chan', 'close', 'wait', 'done'], 'error_handling': ['err', 'error', 'return', 'panic', 'recover'], 'storage_cache': ['cache', 'freezer', 'database', 'sync', 'flush', 'write', 'read'], 'test_change': ['test', 'assert', 'require', 'expected', 'want'], 'generic': []}.get(pattern, [])
    if not keywords:
        return 0.2
    hits = sum((1 for keyword in keywords if keyword in text))
    return min(1.0, hits / 3)

def load_dotenv(path: str='.env') -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
