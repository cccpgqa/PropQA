"""Rebuild and verify constrained candidates from released graph seeds and Git snapshots."""
import argparse
from common import ROOT,read,save
from candidate_expansion import select

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-pairs',type=int,default=0)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    inputs=read(ROOT/'data/statement_inputs.json.gz')
    seeds=read(ROOT/'data/statement_graph_seeds.json.gz')
    ids=list(inputs)[:args.max_pairs or None]
    print({'pairs':len(ids),'requires_git_snapshots':True})
    if not args.execute:return
    for ident in ids:
        chosen=select(inputs[ident]['inference'],seeds[ident])
        expected=inputs[ident]['full']['candidate_ids']
        if set(chosen)!=set(expected):raise ValueError('Candidate mismatch: '+ident)
        print('Verified candidates:',ident)

if __name__=='__main__':main()
