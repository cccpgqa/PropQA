"""Portable artifact I/O and one user-supplied OpenAI-compatible endpoint."""
import gzip
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)

def dotenv():
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key.strip() in {'API_KEY','BASE_URL','MODEL','GITHUB_TOKEN'}:
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))

class API:
    def __init__(self, cache, budget):
        dotenv()
        if not all(os.environ.get(k) for k in ['API_KEY','BASE_URL','MODEL']):
            raise ValueError('Configure API_KEY, BASE_URL and MODEL in your own .env')
        self.model = os.environ['MODEL']
        self.url = os.environ['BASE_URL'].rstrip('/') + '/chat/completions'
        if not self.url.startswith('https://'):
            raise ValueError('Use an HTTPS API endpoint')
        self.cache, self.budget, self.calls = Path(cache), budget, 0

    def call(self, system, payload, validator):
        body = {'model':self.model, 'temperature':0,
                'messages':[{'role':'system','content':system},
                            {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]}
        raw = json.dumps(body,ensure_ascii=False).encode()
        digest = hashlib.sha256(self.url.encode()+raw).hexdigest()
        path = self.cache / (digest + '.json')
        if path.exists():
            answer = read(path)['answer']; validator(answer); return answer
        if self.calls >= self.budget:
            raise RuntimeError('API request budget exhausted; completed responses remain cached')
        self.calls += 1
        req = Request(self.url, data=raw, headers={
            'Content-Type':'application/json', 'Authorization':'Bearer '+os.environ['API_KEY']})
        try:
            with urlopen(req,timeout=180) as response:
                obj = json.load(response)
        except Exception as exc:
            # Never echo headers, request objects, or provider bodies containing credentials.
            raise RuntimeError('API request failed: '+type(exc).__name__) from None
        content = obj['choices'][0]['message']['content'].strip()
        if content.startswith('```'):
            content = '\n'.join(content.splitlines()[1:-1])
        answer = json.loads(content)
        validator(answer)
        save(path, {'model':self.model,'answer':answer,'usage':obj.get('usage',{})})
        return answer
