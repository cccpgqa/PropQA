"""Source-statement cosine retrieval for the graph/QA module controls."""
import math
import re
from collections import Counter, defaultdict
from ast_statement_protocol import extract
from snapshots import git
def terms(text):
    return Counter(re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+', text.lower()))

def lexical(inference):
    target, contents = {}, {}
    for path in sorted({u['path'] for u in inference['target_statements']}):
        text=git(inference['target_repo'],'show',f'{inference["target_base"]}:{path}')
        contents[path]=text
        a=extract(path,text)
        target.update({u['id']:terms(' '.join(a['tokens'][i]['value'] for i in u['token_indices'])) for u in a['units']})
    ids={u['id'] for u in inference['target_statements']}
    target={i:t for i,t in target.items() if i in ids}
    source=inference['source']; src={}
    for path in sorted({u['path'] for u in source['changed_statements']}):
        a=extract(path,git(source['repo'],'show',f'{source["base_sha"]}:{path}'))
        src.update({u['id']:terms(' '.join(a['tokens'][i]['value'] for i in u['token_indices'])) for u in a['units']})
    df=Counter(t for v in target.values() for t in v)
    idf={t:math.log((1+len(target))/(1+n))+1 for t,n in df.items()}
    inverse=defaultdict(list)
    for ident,v in target.items():
        w={t:(1+math.log(n))*idf[t] for t,n in v.items()}; norm=math.sqrt(sum(x*x for x in w.values())) or 1
        for t,x in w.items():inverse[t].append((ident,x/norm))
    scores={i:0.0 for i in target}
    for u in source['changed_statements']:
        w={t:(1+math.log(n))*idf.get(t,math.log(1+len(target))+1) for t,n in src[u['id']].items()}
        norm=math.sqrt(sum(x*x for x in w.values())) or 1
        local=defaultdict(float)
        for t,x in w.items():
            for ident,y in inverse.get(t,[]):local[ident]+=x*y/norm
        for ident,x in local.items():scores[ident]=max(scores[ident],min(1.0,x))
    return scores,contents
