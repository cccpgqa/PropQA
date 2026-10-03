from __future__ import annotations
import urllib.request
from typing import Any
import subprocess
import re
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
import json
import time
import dataclasses
TOKEN_RE = re.compile('[A-Za-z_][A-Za-z0-9_]*|\\d+|[^\\sA-Za-z0-9_]')

@dataclasses.dataclass(frozen=True)
class Side:
    role: str
    project: str
    repo: str
    kind: str
    number: int | str | None
    resolved_from_kind: str | None
    resolved_from_number: int | None
    title: str
    body: str
    message: str
    url: str
    merged_at: str
    base_sha: str
    merged_sha: str
    file_names: list[str]

@dataclasses.dataclass
class PatchStatement:
    path: str
    line_no: int
    kind: str
    text: str

class GitHubClient:

    def __init__(self, cache_dir: Path, token: str | None=None, sleep_seconds: float=0.0):
        self.cache_dir = cache_dir
        self.token = token
        self.sleep_seconds = sleep_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str, suffix: str) -> Path:
        digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{digest}.{suffix}'

    def get_json(self, url: str) -> Any:
        path = self._cache_path(url, 'json')
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        req = urllib.request.Request(url, headers=self._headers('application/vnd.github+json'))
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = resp.read().decode('utf-8')
                break
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        assert payload is not None
        path.write_text(payload, encoding='utf-8')
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return json.loads(payload)

    def get_text(self, url: str) -> str:
        path = self._cache_path(url, 'txt')
        if path.exists():
            return path.read_text(encoding='utf-8', errors='replace')
        req = urllib.request.Request(url, headers=self._headers('text/plain'))
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = resp.read().decode('utf-8', errors='replace')
                break
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        assert payload is not None
        path.write_text(payload, encoding='utf-8')
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return payload

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {'Accept': accept, 'User-Agent': 'propagation-detector-baseline', 'X-GitHub-Api-Version': '2022-11-28'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers

    def compare(self, repo: str, base_sha: str, head_sha: str) -> dict[str, Any]:
        base = urllib.parse.quote(base_sha, safe='')
        head = urllib.parse.quote(head_sha, safe='')
        return self.get_json(f'https://api.github.com/repos/{repo}/compare/{base}...{head}')

    def commit(self, repo: str, sha: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(sha, safe='')
        return self.get_json(f'https://api.github.com/repos/{repo}/commits/{quoted}')

    def pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        page = 1
        while True:
            url = f'https://api.github.com/repos/{repo}/pulls/{number}/files?per_page=100&page={page}'
            chunk = self.get_json(url)
            if not chunk:
                break
            files.extend(chunk)
            if len(chunk) < 100:
                break
            page += 1
        return files

    def tree_paths(self, repo: str, sha: str) -> list[str]:
        data = self.get_json(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1')
        return [item['path'] for item in data.get('tree', []) if item.get('type') == 'blob' and isinstance(item.get('path'), str)]

    def file_at(self, repo: str, sha: str, path: str) -> str | None:
        git_dir = Path('.cache/git_repos') / f"{repo.replace('/', '_')}.git"
        if git_dir.exists():
            try:
                result = subprocess.run(['git', f'--git-dir={git_dir}', 'show', f'{sha}:{path}'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20, check=False)
                if result.returncode == 0:
                    return result.stdout
                return None
            except subprocess.TimeoutExpired:
                pass
        quoted = '/'.join((urllib.parse.quote(part, safe='') for part in path.split('/')))
        url = f'https://raw.githubusercontent.com/{repo}/{sha}/{quoted}'
        try:
            return self.get_text(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

def changed_files(compare_data: dict[str, Any], fallback: list[str]) -> list[dict[str, Any]]:
    files = compare_data.get('files') or []
    if files:
        return files
    return [{'filename': name, 'patch': ''} for name in fallback]

def remote_changed_files(client: GitHubClient, side: Side) -> list[dict[str, Any]]:
    if side.kind == 'pr' and side.number:
        files = client.pull_files(side.repo, side.number)
        if files:
            return files
    if side.kind == 'commit':
        commit_data = client.commit(side.repo, side.merged_sha)
        files = commit_data.get('files') or []
        if files:
            return files
    compare_data = client.compare(side.repo, side.base_sha, side.merged_sha)
    return changed_files(compare_data, side.file_names)

def path_tokens(path: str) -> set[str]:
    parts = re.split('[/_.\\-\\s]+', path.lower())
    return {part for part in parts if part}

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and (not b):
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def path_score(source_path: str, target_path: str) -> float:
    if source_path == target_path:
        return 1.0
    source_name = source_path.rsplit('/', 1)[-1]
    target_name = target_path.rsplit('/', 1)[-1]
    source_ext = source_name.rsplit('.', 1)[-1] if '.' in source_name else ''
    target_ext = target_name.rsplit('.', 1)[-1] if '.' in target_name else ''
    path_ratio = SequenceMatcher(None, source_path.lower(), target_path.lower()).ratio()
    name_ratio = SequenceMatcher(None, source_name.lower(), target_name.lower()).ratio()
    token_ratio = jaccard(path_tokens(source_path), path_tokens(target_path))
    ext_bonus = 1.0 if source_ext and source_ext == target_ext else 0.0
    return 0.42 * path_ratio + 0.3 * name_ratio + 0.18 * token_ratio + 0.1 * ext_bonus

def iter_patch_statements(files: list[dict[str, Any]], include_context: bool=True) -> list[PatchStatement]:
    statements: list[PatchStatement] = []
    hunk_new_line = 0
    hunk_old_line = 0
    for file_info in files:
        path = file_info.get('filename') or file_info.get('previous_filename') or ''
        patch = file_info.get('patch') or ''
        for line in patch.splitlines():
            if line.startswith('@@'):
                match = re.search('-(\\d+)(?:,\\d+)? \\+(\\d+)(?:,\\d+)?', line)
                if match:
                    hunk_old_line = int(match.group(1))
                    hunk_new_line = int(match.group(2))
                continue
            if line.startswith('+++') or line.startswith('---'):
                continue
            if line.startswith('+'):
                text = line[1:].strip()
                if meaningful_statement(text):
                    statements.append(PatchStatement(path, hunk_new_line, 'added', text))
                hunk_new_line += 1
            elif line.startswith('-'):
                text = line[1:].strip()
                if meaningful_statement(text):
                    statements.append(PatchStatement(path, hunk_old_line, 'deleted', text))
                hunk_old_line += 1
            else:
                text = line[1:].strip() if line.startswith(' ') else line.strip()
                if include_context and meaningful_statement(text):
                    statements.append(PatchStatement(path, hunk_old_line, 'context', text))
                hunk_old_line += 1
                hunk_new_line += 1
    return statements

def meaningful_statement(text: str) -> bool:
    if not text or text in {'{', '}', '},', ');'}:
        return False
    if text.startswith('//') or text.startswith('*'):
        return False
    return len(TOKEN_RE.findall(text)) >= 3

def token_similarity(a: str, b: str) -> float:
    a_tokens = set(TOKEN_RE.findall(a.lower()))
    b_tokens = set(TOKEN_RE.findall(b.lower()))
    return 0.55 * SequenceMatcher(None, a.strip(), b.strip()).ratio() + 0.45 * jaccard(a_tokens, b_tokens)
