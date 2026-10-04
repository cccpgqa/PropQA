"""Collect public PR, issue and reachable commit records; no LLM is used."""
import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from common import ROOT, dotenv, read, save

REPOS=('ethereum/go-ethereum','bnb-chain/bsc','0xPolygon/bor','celo-org/celo-blockchain')

def endpoint(repo, suffix, **params):
    if repo not in REPOS:raise ValueError('Repository outside the study scope')
    return 'https://api.github.com/repos/'+repo+'/'+suffix+'?'+urlencode(params)

class GitHub:
    def __init__(self, cache, refresh=False, pause=.2):
        self.cache=Path(cache);self.refresh=refresh;self.pause=pause
        self.token=os.environ.get('GITHUB_TOKEN','');self.requests=0

    def get(self, url):
        parsed=urlsplit(url)
        if parsed.scheme!='https' or parsed.hostname!='api.github.com' or parsed.username:
            raise ValueError('Refusing to send credentials outside api.github.com')
        dest=self.cache/(hashlib.sha256(url.encode()).hexdigest()+'.json')
        if dest.exists() and not self.refresh:return read(dest)
        headers={'Accept':'application/vnd.github+json','User-Agent':'PropQA-replication',
                 'X-GitHub-Api-Version':'2022-11-28'}
        if self.token:headers['Authorization']='Bearer '+self.token
        for attempt in range(3):
            try:
                self.requests+=1
                with urlopen(Request(url,headers=headers),timeout=90) as response:
                    result={'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),
                            'body':json.load(response),'link':response.headers.get('Link','')}
                save(dest,result);time.sleep(self.pause);return result
            except HTTPError as exc:
                if exc.code in (403,429):
                    raise RuntimeError('GitHub rate limit or permission error; stop and resume later') from None
                if exc.code<500 or attempt==2:raise RuntimeError(f'GitHub HTTP {exc.code}') from None
            except (URLError,TimeoutError):
                if attempt==2:raise RuntimeError('GitHub connection failed; cached responses are retained') from None
            time.sleep(2**attempt)

    def pages(self,url,max_pages=0):
        page=0
        while url:
            result=self.get(url);body=result['body']
            if not isinstance(body,list):raise ValueError('Expected a GitHub list response')
            page+=1
            match=re.search(r'<([^>]+)>;\s*rel="next"',result['link'])
            next_url=match.group(1) if match else None
            yield body, bool(next_url and max_pages and page>=max_pages)
            if max_pages and page>=max_pages:return
            url=next_url

    def all(self,url):
        return [item for rows,_ in self.pages(url) for item in rows]

def record(repo, kind, item):
    identity=str(item['sha'] if kind=='commit' else item['number'])
    return {'repo':repo,'kind':kind,'id':identity,'url':item.get('html_url'),'raw':item}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repos',nargs='+',choices=REPOS,default=list(REPOS))
    p.add_argument('--kinds',nargs='+',choices=['pr','issue','commit'],default=['pr','issue','commit'])
    p.add_argument('--output',type=Path,default=ROOT/'raw')
    p.add_argument('--max-pages',type=int,default=0,help='0 is unbounded; applied to each repository/kind/ref')
    p.add_argument('--ref',action='append',help='Commit branch/ref; repeat to cover multiple histories')
    p.add_argument('--include-diffs',action='store_true',help='Fetch PR/commit details and changed files; substantially more requests')
    p.add_argument('--include-discussions',action='store_true',help='Fetch issue comments/timeline and PR conversation comments')
    p.add_argument('--refresh',action='store_true',help='Refresh cached API responses')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.max_pages<0:p.error('--max-pages must be nonnegative')
    print({'repositories':args.repos,'kinds':args.kinds,'commit_refs':args.ref or ['default branch'],
           'max_pages':args.max_pages,'network_enabled':args.execute})
    if not args.execute:return
    dotenv();api=GitHub(args.output/'.responses',args.refresh)
    scopes=[];counts={};seen=set()
    for repo in args.repos:
        for kind in args.kinds:
            refs=(args.ref or [None]) if kind=='commit' else [None]
            for ref in refs:
                params={'per_page':100}
                suffix={'pr':'pulls','issue':'issues','commit':'commits'}[kind]
                if kind in ('pr','issue'):params.update(state='all',sort='created',direction='asc')
                if ref:params['sha']=ref
                scope={'repo':repo,'kind':kind,'ref':ref,'truncated':False,'records':0}
                for rows,truncated in api.pages(endpoint(repo,suffix,**params),args.max_pages):
                    scope['truncated']|=truncated
                    for item in rows:
                        # The issues endpoint also contains PRs; they are collected separately.
                        if kind=='issue' and 'pull_request' in item:continue
                        base=record(repo,kind,item);key=(repo,kind,base['id'])
                        if key in seen:continue
                        seen.add(key)
                        ident=base['id'];dest=args.output/repo.replace('/','_')/kind/(ident+'.json')
                        if kind=='pr' and args.include_diffs:
                            base['raw']=api.get(endpoint(repo,'pulls/'+ident))['body']
                            base['files']=api.all(endpoint(repo,'pulls/'+ident+'/files',per_page=100))
                        elif kind=='commit' and args.include_diffs:
                            url=endpoint(repo,'commits/'+ident,per_page=100);files=[];detail=None
                            while url:
                                response=api.get(url);page=response['body']
                                detail=detail or page;files.extend(page.get('files',[]))
                                match=re.search(r'<([^>]+)>;\s*rel="next"',response['link'])
                                url=match.group(1) if match else None
                            base['raw']=detail;base['files']=files
                        if args.include_discussions and kind in ('pr','issue'):
                            base['comments']=api.all(endpoint(repo,'issues/'+ident+'/comments',per_page=100))
                            if kind=='issue':base['timeline']=api.all(endpoint(repo,'issues/'+ident+'/timeline',per_page=100))
                        base['retrieved_at']=datetime.now(timezone.utc).isoformat()
                        save(dest,base);scope['records']+=1
                    save(args.output/'collection_manifest.json',{'scopes':scopes+[scope],'complete':False,'api_requests_this_run':api.requests})
                scopes.append(scope);counts[repo+':'+kind]=counts.get(repo+':'+kind,0)+scope['records']
                print(repo,kind,scope['records'],'records',flush=True)
    save(args.output/'collection_manifest.json',{'scopes':scopes,'complete':not any(s['truncated'] for s in scopes),
         'scope_note':'Commit coverage is limited to the specified refs, or the current default branch.',
         'api_requests_this_run':api.requests,'counts':counts})

if __name__=='__main__':main()
