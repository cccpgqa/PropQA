"""Check release integrity, benchmark counts and isolation of target labels."""
import ast
import hashlib
from common import ROOT,read

def main():
    manifest=read(ROOT/'data/artifact_manifest.json')
    for name,expected in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Artifact integrity mismatch: '+name)
    assert len(read(ROOT/'data/empirical_pairs.json'))==265
    assert len(read(ROOT/'data/detection_pairs.json'))==224
    file=read(ROOT/'data/file_inputs.json.gz');statement=read(ROOT/'data/statement_inputs.json.gz')
    labels=read(ROOT/'data/statement_labels.json.gz')
    assert len(file)==224 and len(statement)==len(labels)==184
    assert sum(len(c['units']) for c in labels)==168593
    assert sum(u['label'] for c in labels for u in c['units'])==3570
    assert set(statement)=={c['pair_id'] for c in labels}
    for ident,item in statement.items():
        census={u['id'] for u in item['inference']['target_statements']}
        assert all('label' not in u for u in item['inference']['target_statements'])
        for variant in ['full','without_graph']:
            plan=item[variant];chosen=set(plan['candidate_ids'])
            assert chosen<=census
            assert {i for b in plan['batches'] for i in b['ids']}==chosen
            assert len(chosen)<=360
    cases=read(ROOT/'data/practical_propagation_cases_14.json')
    assert [c['case_id'] for c in cases]==[f'CASE-{i:03d}' for i in range(1,15)]
    for p in (ROOT/'scripts').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    assert not (ROOT/'.env').exists(), 'Do not include private .env in the release'
    print('Integrity, syntax, cohorts and inference/label separation passed.')

if __name__=='__main__':main()
