"""Offline tests: no credentials, network, repository checkout or API calls."""
import unittest
from common import ROOT,read
from evaluate import files
from file_answers import validate
from statement_context import validate_answer
from statement_binary_evaluation import fit_threshold,evaluate,grouped_folds
from run_localization import file_predict,statement_predict

class FakeAPI:
    model='offline-test'
    def call(self,system,payload,validator):
        if 'file_queries' in payload:
            rows=[]
            for q in payload['file_queries']:
                if 'action_questions' in payload:
                    row={'query_id':q['query_id']}
                    for k in payload['action_questions']:row[k]=[]
                    rows.append(row)
                else:
                    rows.append({'query_id':q['query_id'],'ranked_candidate_ids':list(range(min(5,len(q['candidate_target_files']))))})
            answer={'action_answers' if 'action_questions' in payload else 'rankings':rows}
        else:
            answer={'complete':True,'reviewed_count':len(payload['target']['nodes']),
                    'affected':[],'evidence':'offline test'}
        validator(answer)
        return answer

class Tests(unittest.TestCase):
    def test_all_file_plans(self):
        for item in read(ROOT/'data/file_inputs.json.gz').values():
            for variant in ['full','without_graph']:
                self.assertTrue(file_predict(FakeAPI(),item,variant)['complete'])
    def test_all_statement_plans(self):
        for item in read(ROOT/'data/statement_inputs.json.gz').values():
            for variant in ['full','without_graph']:
                result=statement_predict(FakeAPI(),item,variant)
                self.assertEqual(set(result['scores']),{u['id'] for u in item['inference']['target_statements']})
    def test_reject_duplicate_statement(self):
        with self.assertRaises(ValueError):
            validate_answer({'complete':True,'reviewed_count':2,'affected':[[0,.9,'edit'],[0,.8,'delete']],'evidence':''},2)
    def test_file_metrics(self):
        self.assertEqual(files(read(ROOT/'results/file_reference_ranks.json')),read(ROOT/'results/file_metrics.json'))
    def test_threshold(self):
        cases=[{'pair_id':'1','units':[{'label':1,'scores':{'m':.8}}, {'label':0,'scores':{'m':.2}}, {'label':1,'scores':{'m':None}}]}]
        p=fit_threshold(cases,'m')['p']
        self.assertLess(p,.8);self.assertGreater(p,.2)
        r=evaluate(cases,'m',p)[0]
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,0,1))

if __name__=='__main__':unittest.main()
