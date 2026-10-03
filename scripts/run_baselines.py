"""Recompute native NiCad-Go and Go code2vec on frozen repository snapshots."""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
from common import ROOT,read,save
from snapshots import code

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level',choices=['file','statement'],required=True)
    p.add_argument('--method',choices=['nicad','code2vec'],required=True)
    p.add_argument('--path-prefilter',action='store_true')
    p.add_argument('--engine',type=Path,help='Built official crossclones executable')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-pairs',type=int,default=0)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.path_prefilter and args.level=='statement':raise ValueError('Path prefilter is a file-level variant')
    dataset=read(ROOT/f'data/{args.level}_inputs.json.gz')
    ids=list(dataset)[:args.max_pairs or None]
    print({'level':args.level,'method':args.method,'pairs':len(ids),'requires_git_snapshots':True})
    if not args.execute:return
    if args.method=='nicad':
        from nicad_go import extract, compare
        from clone_scores import score_nicad
        if not args.engine or not args.engine.is_file():raise ValueError('Supply --engine; see docs/BASELINES.md')
        args.engine=args.engine.resolve()
    else:
        import numpy as np
        import torch
        from go_code2vec import GoCode2VecEncoder
        from statement_vectors import encode_statements
        torch.set_num_threads(2)
        encoder=GoCode2VecEncoder(ROOT/'models/go_code2vec.pt')
    name=('NiCad-Go' if args.method=='nicad' else 'code2vec')+('+Path' if args.path_prefilter else '')
    manifest={'level':args.level,'variant':name,'pair_ids':ids,'model':None}
    if (args.output/'manifest.json').exists() and read(args.output/'manifest.json')!=manifest:raise ValueError('Output configuration changed')
    save(args.output/'manifest.json',manifest)
    for ident in ids:
        out=args.output/f'pair_{int(ident):04d}.json'
        if out.exists():continue
        item=dataset[ident]
        if args.level=='file':
            target={p:code(item['target_repo'],item['target_base'],p) for p in item['target_tree'] if p.endswith('.go')}
            source={s['path']:code(s['repo'],s['revision'],s['actual_path']) for s in item['sources'] if s['actual_path'].endswith('.go') and s.get('kind')!='gitlink'}
            scores={s['path']:{} for s in item['sources']}
            if args.method=='nicad':
                def supported(text,path):
                    try:return extract(text,path)
                    except ValueError:return []
                sf=[f for path,t in source.items() for f in supported(t,path)]
                tf=[f for path,t in target.items() for f in supported(t,path)]
                root=compare(args.engine,sf,tf,args.output/f'engine_{ident}') if sf and tf else ET.Element('clones')
                for clone in root.iter('clone'):
                    ends={x.attrib['file'].split('/',1)[0]:x.attrib['file'].split('/',1)[1] for x in clone.findall('source')}
                    a,b=ends['source'],ends['target']
                    scores[a][b]=max(scores[a].get(b,0),float(clone.attrib['similarity'])/100)
            else:
                tv={p:v for p,t in target.items() if (v:=encoder.file_vector(t)) is not None}
                for path,t in source.items():
                    v=encoder.file_vector(t)
                    if v is not None:scores[path]={p:float(w@v) for p,w in tv.items()}
            predictions={}
            for path,values in scores.items():
                order=sorted(values,key=lambda p:(-values[p],p))
                if args.path_prefilter:order=[p for p in order if p in set(item['path_masks'][path])]
                predictions[path]=order[:5]
            result={'predictions':predictions}
        else:
            inf=item['inference']; units=inf['target_statements'];src=inf['source']
            source={p:code(src['repo'],src['base_sha'],p) for p in sorted({u['path'] for u in src['changed_statements']})}
            target={p:code(inf['target_repo'],inf['target_base'],p) for p in sorted({u['path'] for u in units})}
            if args.method=='nicad':
                sf=[f for path,t in source.items() for f in extract(t,path)]
                tf=[f for path,t in target.items() for f in extract(t,path)]
                work=args.output/f'engine_{ident}'
                if sf and tf:
                    compare(args.engine,sf,tf,work);scores=score_nicad(work/'clones.xml',units)
                else:scores={u['id']:None for u in units}
            else:
                sv=[v for path,t in source.items() for v in encode_statements(encoder,t,[u for u in src['changed_statements'] if u['path']==path])]
                tv=[v for path,t in target.items() for v in encode_statements(encoder,t,[u for u in units if u['path']==path])]
                scores={u['id']:None for u in units}
                if sv and tv:
                    matrix=np.stack([v['vector'] for v in sv])@np.stack([v['vector'] for v in tv]).T
                    scores.update({v['id']:round(max(0.,min(1.,float(matrix[:,i].max()))),6) for i,v in enumerate(tv)})
            result={'scores':scores}
        save(out,{'pair_id':ident,'complete':True,**result})
        print('Completed pair',ident,flush=True)

if __name__=='__main__':main()
