"""Offline collection and filtering checks using synthetic GitHub records."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collect_github import GitHub, endpoint, record
from filter_candidates import references, exclusion, resolution_candidates, main
from common import save, read


class CollectionTests(unittest.TestCase):
    def test_url_identity(self):
        text='https://github.com/ethereum/go-ethereum/pull/12 https://github.com/ethereum/go-ethereum/pull/12'
        self.assertEqual(list(references(text)),[('ethereum/go-ethereum','pr','12')])
        self.assertEqual(list(references('https://github.com/other/repo/pull/12')),[])

    def test_pagination(self):
        api=GitHub('unused-cache')
        responses=[{'body':[1],'link':'<https://api.github.com/page2>; rel="next"'},
                   {'body':[2],'link':''}]
        with patch.object(api,'get',side_effect=responses):
            self.assertEqual(list(api.pages('https://api.github.com/page1')), [([1],False),([2],False)])
        with patch.object(api,'get',return_value=responses[0]):
            self.assertEqual(list(api.pages('https://api.github.com/page1',1)),[([1],True)])

    def test_credential_host_guard(self):
        with self.assertRaises(ValueError):GitHub('unused').get('https://example.org')
        with self.assertRaises(ValueError):endpoint('other/repo','pulls')

    def test_merge_and_revert(self):
        source={'repo':'ethereum/go-ethereum','kind':'pr','state':'closed','merged':False}
        target={'repo':'bnb-chain/bsc','kind':'pr','title':'Fix','merged':True}
        self.assertEqual(exclusion(source,target),'source_unmerged_pr')
        source.update(merged=True,merged_at='2024-01-01T00:00:00Z')
        self.assertIsNone(exclusion(source,target))
        target['title']='Revert "Fix"'
        self.assertEqual(exclusion(source,target),'target_revert')

    def test_issue_reference_not_proof(self):
        candidates=resolution_candidates({'timeline':[{'event':'cross-referenced','source':{
            'issue':{'html_url':'https://github.com/ethereum/go-ethereum/pull/12'}}}]})
        self.assertTrue(candidates[0]['requires_confirmation'])

    def test_filter_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';out=root/'out'
            source=record('ethereum/go-ethereum','pr',{'number':12,'title':'Fix','state':'closed',
                'merged':True,'merged_at':'2024-01-01T00:00:00Z','created_at':'2024-01-01T00:00:00Z'})
            target=record('bnb-chain/bsc','pr',{'number':34,'title':'Fix','merged':True,
                'merged_at':'2024-02-01T00:00:00Z','created_at':'2024-02-01T00:00:00Z',
                'body':'From https://github.com/ethereum/go-ethereum/pull/12'})
            for item in [source,target]:
                save(raw/item['repo'].replace('/','_')/item['kind']/(item['id']+'.json'),item)
            with patch('sys.argv',['filter_candidates','--raw',str(raw),'--output',str(out)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            pairs=read(out/'candidates.json')
            self.assertEqual(len(pairs),1)
            self.assertEqual((pairs[0]['source']['id'],pairs[0]['target']['id']),('12','34'))
            self.assertEqual(pairs[0]['status'],'requires_validation')


if __name__=='__main__':unittest.main()
