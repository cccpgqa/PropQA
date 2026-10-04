"""Offline URL pairing and initial exclusion rules, not propagation validation."""
import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from common import ROOT,read,save
from collect_github import REPOS

CANONICAL={r.lower():r for r in REPOS}
URL=re.compile(r'https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/(pull|issues|commit)/([A-Za-z0-9]+)',re.I)

def references(text):
    seen=set()
    for m in URL.finditer(text):
        repo=CANONICAL.get(m[1].lower());kind={'pull':'pr','issues':'issue','commit':'commit'}[m[2].lower()]
        ident=m[3].lower()
        if not repo or (kind!='commit' and not ident.isdecimal()):continue
        if kind=='commit' and not re.fullmatch(r'[a-f0-9]{7,40}',ident):continue
        key=(repo,kind,ident)
        if key not in seen:seen.add(key);yield key

def identity(record):
    raw=record['raw'];kind=record['kind'];repo=record['repo']
    commit=raw.get('commit') or {}
    return {'repo':repo,'kind':kind,'id':str(record['id']),
            'url':record.get('url') or raw.get('html_url'),
            'title':raw.get('title',''),'body':raw.get('body') or '',
            'message':commit.get('message','') if kind=='commit' else '',
            'created_at':raw.get('created_at') or (commit.get('committer') or {}).get('date'),
            'merged_at':raw.get('merged_at'),'merged':raw.get('merged'),
            'state':raw.get('state'),
            'files':[x['filename'] for x in record.get('files',[]) if x.get('filename')],
            'has_file_details':'files' in record}

def exclusion(source,target,max_files=0):
    if source['repo']==target['repo']:return 'same_repository'
    for label,side in [('source',source),('target',target)]:
        if side['kind']=='pr' and (side.get('merged') is False or
            (side.get('state')=='closed' and not side.get('merged_at'))):return label+'_unmerged_pr'
    title=target.get('title') or target.get('message','').split('\n')[0]
    if re.match(r'^\s*revert\b',title,re.I):return 'target_revert'
    if max_files and any(len(s.get('files',[]))>max_files for s in [source,target]):return 'file_count_limit'
    return None

def resolution_candidates(record):
    out=[]
    for event in record.get('timeline',[]):
        nested=(event.get('source') or {}).get('issue') or {}
        urls=[nested.get('html_url',''),event.get('commit_url','')]
        # Cross-references alone do not prove that a change resolved the issue.
        for url in urls:
            if url.startswith('https://api.github.com/repos/'):
                url=url.replace('https://api.github.com/repos/','https://github.com/').replace('/commits/','/commit/')
            for repo,kind,ident in references(url):
                if kind in ('pr','commit'):
                    item={'repo':repo,'kind':kind,'id':ident,'event':event.get('event'),'requires_confirmation':True}
                    if item not in out:out.append(item)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'raw')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/mining')
    p.add_argument('--max-changed-files',type=int,default=0)
    args=p.parse_args()
    if args.max_changed_files<0:p.error('File limit must be nonnegative')
    paths=sorted(p for repo in REPOS for kind in ['pr','issue','commit']
                 for p in (args.raw/repo.replace('/','_')/kind).glob('*.json'))
    if not paths:raise ValueError('No raw records found; collect records first')
    args.output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE records(repo TEXT,kind TEXT,id TEXT,path TEXT,PRIMARY KEY(repo,kind,id))')
    for path in paths:
        rec=read(path)
        db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',(rec['repo'],rec['kind'],str(rec['id']),str(path)))
    retained=[];removed=[];seen=set()
    for path in paths:
        rec=read(path);target=identity(rec)
        for repo,kind,ident in references('\n'.join(target.get(k,'') or '' for k in ['title','body','message'])):
            found=db.execute('SELECT path FROM records WHERE repo=? AND kind=? AND id=?',(repo,kind,ident)).fetchall()
            if not found and kind=='commit':found=db.execute('SELECT path FROM records WHERE repo=? AND kind=? AND id LIKE ?',(repo,kind,ident+'%')).fetchall()
            source_record=read(Path(found[0][0])) if len(found)==1 else None
            source=identity(source_record) if source_record else {'repo':repo,'kind':kind,'id':ident,
                'url':f'https://github.com/{repo}/'+{'pr':'pull','issue':'issues','commit':'commit'}[kind]+'/'+ident}
            key=(source['repo'],source['kind'],source['id'],target['repo'],target['kind'],target['id'])
            if key in seen:continue
            seen.add(key)
            entry={'source':source,'target':target,'status':'requires_validation'}
            reason=exclusion(source,target,args.max_changed_files)
            if reason:removed.append({**entry,'reason':reason});continue
            checks=[]
            if source_record is None:checks.append('source_metadata_missing_or_ambiguous')
            for label,side,raw in [('source',source,source_record),('target',target,rec)]:
                if side['kind']=='pr' and not side.get('merged_at'):checks.append(label+'_merge_status_unverified')
                if not side.get('has_file_details'):checks.append(label+'_patch_details_missing')
                if side['kind']=='issue':
                    entry[label+'_resolution_candidates']=resolution_candidates(raw or {})
                    checks.append(label+'_issue_resolution_required')
            try:
                s=datetime.fromisoformat(source['created_at'].replace('Z','+00:00'))
                t=datetime.fromisoformat(target['created_at'].replace('Z','+00:00'))
                if t<s:checks.append('target_timestamp_precedes_source_check_import_history')
            except (KeyError,AttributeError,ValueError):checks.append('timestamps_incomplete')
            entry['checks']=checks;entry['candidate_id']=f'CAND-{len(retained)+1:05d}'
            retained.append(entry)
    db.close()
    save(args.output/'candidates.json',retained);save(args.output/'excluded_candidates.json',removed)
    summary={'raw_records':len(paths),'candidate_pairs':len(retained),'initially_excluded':len(removed),
             'note':'Candidates are not validated propagation pairs. Inspect both patches and resolve issues before acceptance.'}
    save(args.output/'summary.json',summary);print(summary)

if __name__=='__main__':main()
