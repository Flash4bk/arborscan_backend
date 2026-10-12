"""Allowlisted DEMO quarantine fixtures; these tests do not mutate VPS data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

OPS = Path(__file__).resolve().parents[1] / 'deploy-vps'
sys.path.insert(0, str(OPS))
import ops_selfhost_quarantine as quarantine


class FakeHTTP:
    def __init__(self, state, records, sql):
        self.state=state;self.records=copy.deepcopy(records);self.sql=copy.deepcopy(sql)
        self.writes=[];self.stale={}

    def request(self, method, path, **kwargs):
        if path=='/rest/v1/contour_revisions':data=self.sql
        elif '/object/list/' in path:
            if method!='POST':raise AssertionError('only Storage list POST is expected')
            data=[{'name':key} for key in self.records]
        elif method=='GET':
            key=path.rsplit('/',1)[1]
            data=self.stale.pop(key,self.records[key])
        else:raise AssertionError('unexpected write')
        return SimpleNamespace(json=lambda:copy.deepcopy(data))

    def api(self, method, path, token):
        key=path.rsplit('/',1)[1];record=copy.deepcopy(self.records[key])
        record['review_status']=dict(self.state['contours']).get(key,record['review_status'])
        return record

    def put(self, http, path, record):
        key=path.rsplit('/',1)[1]
        self.writes.append(key)
        self.records[key]={**copy.deepcopy(record),'purpose':quarantine.PURPOSE}


class QuarantineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.owner='00000000-0000-4000-8000-000000000001'
        self.keys=['00000000-0000-4000-8000-000000000003_'+str(i)*64+'.json' for i in range(1,5)]
        self.state={'format':1,'tag':quarantine.audit.FIXTURE_NAME,'complete':True,'image_sha256':'a'*64,
            'accounts':[{'id':self.owner,'email':'demo-selfhost-'+'a'*32+'@invalid.test',
                'name':quarantine.audit.FIXTURE_NAME,'password':'synthetic-password','token':'synthetic-token'},
                {'id':'00000000-0000-4000-8000-000000000002','email':'demo-selfhost-'+'b'*32+'@invalid.test',
                'name':quarantine.audit.FIXTURE_NAME,'password':'synthetic-password','token':'synthetic-token'}],
            'contours':[[k,s] for k,s in zip(self.keys[:3],('rejected','accepted','draft'))]}
        self.records={k:{'schema_version':2 if i<3 else 1,'owner_id':self.owner,'image_sha256':'a'*64,
            'mask_sha256':'b'*64,'original_image_base64':'unchanged-image','mask_png_base64':'unchanged-png',
            'editor_state':{'points':[{'x':.1,'y':.3}]},'height_m':19.59,'review_status':'draft' if i<3 else 'pending_review'}
            for i,k in enumerate(self.keys)}
        self.sql=[{'owner_id':self.owner,'correction_id':k,'image_sha256':'a'*64,'status':s,
            'decisions':[{'action':s,'actor_id':'synthetic-admin','at':'2026-10-12'}]} for k,s in self.state['contours']]
        self.baseline={'storage':[]};self.proof={'source_user_matches':0,'account_ids_sha256':'c'*64}
        self.http=FakeHTTP(self.state,self.records,self.sql)

    def apply(self,**kwargs):
        return quarantine.apply_markers(self.root,self.http,'private-corrections',self.state,
                                       self.baseline,self.proof,put=self.http.put,pause=lambda _:None,**kwargs)

    def test_marks_exact_own_four_records_and_preserves_every_other_field_and_sql_decision(self):
        result=self.apply()
        self.assertEqual(result['objects_marked'],4)
        self.assertEqual(set(self.http.writes),set(self.keys))
        self.assertEqual(self.http.sql,self.sql)
        for key,value in self.http.records.items():
            self.assertEqual(value['purpose'],quarantine.PURPOSE)
            self.assertEqual(quarantine.unmarked(value),self.records[key])
        snapshot=json.loads((self.root/'quarantine.originals.private.json').read_text())
        self.assertEqual(snapshot['records'],self.records)
        self.assertFalse(result['training_started'])
        self.assertFalse(result['real_user_records_changed'])

    def test_retry_reuses_original_snapshot_and_never_rewrites_already_marked_records(self):
        first=self.apply();self.http.writes.clear()
        second=self.apply()
        self.assertEqual(first['snapshot_sha256'],second['snapshot_sha256'])
        self.assertEqual(self.http.writes,[])

    def test_partial_failure_keeps_snapshot_and_retry_completes_only_remaining_keys(self):
        original_put=self.http.put
        def interrupted(http,path,record):
            original_put(http,path,record)
            if len(self.http.writes)==1:raise RuntimeError('after marker effect')
        with self.assertRaises(RuntimeError):
            quarantine.apply_markers(self.root,self.http,'private-corrections',self.state,
                self.baseline,self.proof,put=interrupted,pause=lambda _:None)
        self.assertTrue((self.root/'quarantine.originals.private.json').exists())
        first=self.http.writes[0];self.http.writes.clear();self.apply()
        self.assertNotIn(first,self.http.writes)
        self.assertEqual(len(self.http.writes),3)

    def test_wrong_owner_changed_photo_or_existing_real_namespace_refuses_before_any_write(self):
        for mutation in ('owner','image','source-namespace','extra-object'):
            with self.subTest(mutation=mutation):
                self.http=FakeHTTP(self.state,self.records,self.sql);self.baseline={'storage':[]}
                if mutation=='owner':self.http.records[self.keys[0]]['owner_id']=self.state['accounts'][1]['id']
                elif mutation=='image':self.http.records[self.keys[0]]['image_sha256']='e'*64
                elif mutation=='source-namespace':self.baseline['storage']=[{'bucket':'private-corrections','name':'v4-corrections/'+self.owner+'/'+self.keys[0]}]
                else:self.http.records['unexpected']=self.records[self.keys[0]]
                with self.assertRaises(quarantine.QuarantineFailure):self.apply()
                self.assertEqual(self.http.writes,[])

    def test_incomplete_fixture_or_unexpected_decision_refuses_before_any_write(self):
        self.state['complete']=False
        with self.assertRaises(quarantine.QuarantineFailure):self.apply()
        self.state['complete']=True;self.http.sql[0]['status']='accepted'
        with self.assertRaises(quarantine.QuarantineFailure):self.apply()
        self.assertEqual(self.http.writes,[])

    def test_actual_marker_write_readback_refuses_any_other_changed_field(self):
        def tamper(http,path,record):
            self.http.put(http,path,record)
            self.http.records[path.rsplit('/',1)[1]]['height_m']=1
        with self.assertRaisesRegex(quarantine.QuarantineFailure,'beyond_purpose'):
            quarantine.apply_markers(self.root,self.http,'private-corrections',self.state,
                self.baseline,self.proof,put=tamper,pause=lambda _:None)
        self.assertTrue((self.root/'quarantine.originals.private.json').exists())

    def test_delayed_storage_readback_retries_without_extra_write(self):
        self.http.stale[self.keys[0]]=self.records[self.keys[0]]
        result=self.apply()
        self.assertEqual(result['objects_marked'],4)
        self.assertEqual(len(self.http.writes),4)

    def test_direct_source_proof_is_readonly_and_existing_owner_refuses(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append((command,kwargs))
            return SimpleNamespace(returncode=0,stdout=b'0\n')
        proof=quarantine.source_accounts_absent(self.state,execute)
        self.assertEqual(proof['source_user_matches'],0)
        query=calls[0][1]['input']
        self.assertIn(b'REPEATABLE READ READ ONLY',query)
        self.assertNotIn(b'UPDATE',query)
        self.assertNotIn(b'DELETE',query)
        self.assertIn('service=arborscan_backup',calls[0][0])
        with self.assertRaises(quarantine.QuarantineFailure):
            quarantine.source_accounts_absent(self.state,lambda *a,**kw:SimpleNamespace(returncode=0,stdout=b'1\n'))

    def test_real_legacy_and_source_baseline_objects_are_not_added_to_allowlist(self):
        self.http.records[self.keys[3]]['schema_version']=2
        with self.assertRaises(quarantine.QuarantineFailure):self.apply()
        self.assertEqual(self.http.writes,[])

    def test_exporter_probe_calls_existing_sample_both_modes_without_training_or_jobs(self):
        self.assertIn('from arborscan_v4.quality_dataset import sample',quarantine.EXPORTER_PROBE)
        self.assertIn("('segmentation','classification')",quarantine.EXPORTER_PROBE)
        self.assertIn('synthetic_export_smoke_not_evaluation_data',quarantine.EXPORTER_PROBE)
        self.assertNotIn('create_job',quarantine.EXPORTER_PROBE)
        self.assertNotIn('dataset_build',quarantine.EXPORTER_PROBE)


if __name__=='__main__':unittest.main()
