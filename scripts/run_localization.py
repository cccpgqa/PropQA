"""Run final file QA or exact-statement QA on published, label-free evidence."""
import argparse
import copy
import hashlib
from pathlib import Path
from common import ROOT, API, read, save
from file_answers import validate
from statement_context import validate_answer

def file_predict(api, item, variant):
    if variant == 'without_qa':
        import re
        from collections import Counter
        from snapshots import code
        from lexical_retrieval import cosine_rank
        def tokens(text):
            return Counter(x.lower() for x in re.findall(r'[A-Za-z_][A-Za-z_0-9]*|\d+',text))
        target={}
        for path in item['target_tree']:
            try:text=code(item['target_repo'],item['target_base'],path)
            except UnicodeDecodeError:text=''
            target[path]={'tokens':tokens('' if '\0' in text else text)}
        predictions={}
        for source in item['sources']:
            if source.get('kind')=='gitlink':text='Subproject commit '+source['blob']
            else:
                try:text=code(source['repo'],source['revision'],source['actual_path'])
                except UnicodeDecodeError:text=''
            predictions[source['path']]=[r['path'] for r in cosine_rank(tokens(text),target)[:5]]
        return {'complete':True,'predictions':predictions}
    predictions, trace = {}, []
    for batch in item['plans'][variant]:
        answers = []
        for stage, call in enumerate(batch['calls']):
            payload = copy.deepcopy(call['payload'])
            if stage == 1:
                payload['action_answers'] = answers[0]['action_answers']
                payload['constraints'] += [
                    'This is a ranking task, not a binary affected-file list. Return at least '
                    'min(5, number of supplied candidates) distinct ranked_candidate_ids for EVERY '
                    'query, even when only one file is strongly relevant. Order less likely files last.']
                payload['output_schema'] = {'rankings':[{'query_id':0,
                    'ranked_candidate_ids':[0,1,2,3,4],
                    'brief_rationale':'One short evidence-based sentence'}]}
            answers.append(api.call(call['system'],payload,lambda a:validate(a,batch,stage)))
        for row in answers[1]['rankings']:
            query = batch['queries'][row['query_id']]
            predictions[query['source_file']] = [query['candidate_paths'][i] for i in row['ranked_candidate_ids'][:5]]
        trace.append(answers)
    return {'complete':True,'predictions':predictions,'answers':trace,'model':api.model}

def statement_predict(api, item, variant):
    if variant == 'without_qa':
        from statement_retrieval import lexical
        scores, _ = lexical(item['inference'])
        return {'complete':True,'scores':scores}
    plan = item[variant]
    scores = {u['id']:None for u in item['inference']['target_statements']}
    answers=[]
    system=read(ROOT/'data/statement_prompt.json')['system']
    for batch in plan['batches']:
        answer=api.call(system,batch['payload'],lambda a:validate_answer(a,len(batch['ids'])))
        for ident in batch['ids']:scores[ident]=max(scores[ident] or 0,0)
        for i, confidence, operation in answer['affected']:
            ident=batch['ids'][i];scores[ident]=max(scores[ident] or 0,confidence)
        answers.append(answer)
    return {'complete':True,'scores':scores,'answers':answers,'model':api.model}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level',choices=['file','statement'],required=True)
    p.add_argument('--variant',choices=['full','without_graph','without_qa'],default='full')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-pairs',type=int,default=0)
    p.add_argument('--max-requests',type=int,default=1000)
    p.add_argument('--execute',action='store_true',help='Allow API calls; otherwise print a plan only')
    args=p.parse_args()
    inp=ROOT/f'data/{args.level}_inputs.json.gz'
    data=read(inp); ids=list(data)[:args.max_pairs or None]
    count=0 if args.variant=='without_qa' else sum(
        sum(len(b['calls']) for b in data[i]['plans'][args.variant]) if args.level=='file'
        else len(data[i][args.variant]['batches']) for i in ids)
    print({'level':args.level,'variant':args.variant,'pairs':len(ids),'planned_requests':count})
    if not args.execute:return
    if count>args.max_requests:raise ValueError('Planned calls exceed --max-requests')
    api=None if args.variant=='without_qa' else API(args.output/'api_cache',args.max_requests)
    manifest={'level':args.level,'variant':args.variant,'pair_ids':ids,
              'model':api.model if api else None,
              'endpoint_hash':hashlib.sha256(api.url.encode()).hexdigest() if api else None,
              'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest()}
    dest=args.output/'manifest.json'
    if dest.exists() and read(dest)!=manifest:raise ValueError('Use a new output directory for a different configuration')
    save(dest,manifest)
    for ident in ids:
        out=args.output/f'pair_{int(ident):04d}.json'
        if out.exists():
            if read(out).get('complete') is not True:raise ValueError('Incomplete saved prediction')
            continue
        fn=file_predict if args.level=='file' else statement_predict
        save(out,{'pair_id':ident,**fn(api,data[ident],args.variant)})
        print('Completed pair',ident,flush=True)

if __name__=='__main__':main()
