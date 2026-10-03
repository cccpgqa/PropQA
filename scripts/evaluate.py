"""Recompute the published tables, or evaluate a complete fresh inference run."""
import argparse
from pathlib import Path
from common import ROOT, read, save
from statement_binary_evaluation import run

def files(rows):
    methods=list(rows[0]['ranks']);result={}
    for m in methods:
        ranks=[r['ranks'][m] for r in rows];n=len(ranks)
        result[m]={f'Hit@{k}':sum(r is not None and r<=k for r in ranks)/n for k in (1,3,5)}
        result[m]['MRR@5']=sum(1/r for r in ranks if r is not None and r<=5)/n
    return {'pairs':len({r['pair_index'] for r in rows}),'file_pairs':len(rows),'methods':result}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced')
    p.add_argument('--predictions',type=Path)
    p.add_argument('--name',default='PropQA-user-model')
    args=p.parse_args()
    if args.predictions:
        manifest=read(args.predictions/'manifest.json');level=manifest['level']
        inp=read(ROOT/f'data/{level}_inputs.json.gz')
        if set(manifest['pair_ids'])!=set(inp):raise ValueError('Full benchmark predictions are required for the published evaluation')
        pred={i:read(args.predictions/f'pair_{int(i):04d}.json') for i in inp}
        if not all(x['complete'] for x in pred.values()):raise ValueError('Incomplete inference')
        if level=='file':
            rows=[]
            for row in read(ROOT/'results/file_reference_ranks.json'):
                paths=pred[str(row['pair_index'])]['predictions'][row['source_file']]
                rank=paths.index(row['target_file'])+1 if row['target_file'] in paths else None
                rows.append({**row,'ranks':{args.name:rank}})
            result=files(rows)
        else:
            cases=read(ROOT/'data/statement_labels.json.gz')
            for c in cases:
                values=pred[c['pair_id']]['scores']
                if set(values)!={u['id'] for u in c['units']}:raise ValueError('Incomplete AST census')
                for u in c['units']:u['scores']={args.name:values[u['id']]}
            result=run({'methods':[args.name],'cases':cases},5,2027)
        save(args.output/(level+'_metrics.json'),result)
        return
    fm=files(read(ROOT/'results/file_reference_ranks.json'))
    sm=run(read(ROOT/'results/statement_scores.json.gz'),5,2027)
    expected=read(ROOT/'results/file_metrics.json')
    if fm!=expected:raise ValueError('File metrics differ from published artifacts')
    expected=read(ROOT/'results/statement_metrics.json')
    if sm['folds']!=expected['folds']:raise ValueError('Fold assignment mismatch')
    for m,r in sm['methods'].items():
        if r['micro']!=expected['methods'][m]['micro']:raise ValueError('Statement metric mismatch: '+m)
    save(args.output/'file_metrics.json',fm);save(args.output/'statement_metrics.json',sm)
    text=['# Reproduced Results','','## File-Level','','| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |','|---|---:|---:|---:|---:|']
    text += ['| '+m+' | '+' | '.join(f'{r[k]:.3f}' for k in ['Hit@1','Hit@3','Hit@5','MRR@5'])+' |' for m,r in fm['methods'].items()]
    text+=['','## Statement-Level','','| Method | Precision | Recall | F1 | MCC |','|---|---:|---:|---:|---:|']
    text+=['| '+m+' | '+' | '.join(f'{r["micro"][k]:.3f}' for k in ['precision','recall','f1','mcc'])+' |' for m,r in sm['methods'].items()]
    (args.output/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('All published metrics reproduced exactly from per-unit scores and reference ranks.')

if __name__=='__main__':main()
