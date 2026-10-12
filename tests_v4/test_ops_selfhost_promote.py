"""Promotion guard tests: no production container is used by these tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

OPS = Path(__file__).resolve().parents[1] / 'deploy-vps'
sys.path.insert(0, str(OPS))
import ops_selfhost_promote as promote


class PromotionSafetyTests(unittest.TestCase):
    def config(self):
        services = {key: {'image': 'sha256:' + '1' * 64,
            'environment': {'SUPABASE_URL': 'http://gateway:8000', 'GOOGLE_POLICY': 'retained'},
            'volumes': [], 'cpus': 1} for key in ('db', 'rest', 'storage', 'gateway', 'v3', 'api', 'worker')}
        services['v3']['ports'] = ['127.0.0.1:19000:8000']
        services['api']['ports'] = ['127.0.0.1:19001:8001']
        services['worker'].update(profiles=['manual-worker'], restart='no')
        return {'name': 'selfhost-test', 'services': services,
            'networks': {'default': {'internal': True}, 'providers': {}}, 'volumes': {'db': {}, 'storage': {}}}

    def test_promotion_changes_only_reviewed_api_fields_and_never_db_or_keys(self):
        original = self.config()
        source = {key: {'Image': original['services'][key]['image']} for key in ('v3', 'api')}
        changed = promote.plan(original, source)
        self.assertEqual(original, self.config())
        restored = copy.deepcopy(changed)
        for service in ('v3', 'api'):
            item = restored['services'][service]
            item['ports'] = original['services'][service]['ports']
            item['cpus'] = original['services'][service]['cpus']
            del item['container_name']
        self.assertEqual(restored, original)
        self.assertEqual(changed['services']['api']['ports'], ['127.0.0.1:8001:8001'])
        self.assertEqual(changed['services']['worker']['restart'], 'no')

    def test_public_db_or_changed_image_or_automatic_worker_refused(self):
        for mutation in ('network', 'image', 'worker'):
            with self.subTest(mutation=mutation):
                config = self.config()
                source = {key: {'Image': config['services'][key]['image']} for key in ('v3', 'api')}
                if mutation == 'network': config['networks']['default']['internal'] = False
                elif mutation == 'image': source['api']['Image'] = 'sha256:' + '2' * 64
                else: config['services']['worker']['restart'] = 'always'
                with self.assertRaises(promote.PromotionFailure): promote.plan(config, source)

    def test_changed_local_data_prevents_any_activation_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = b'{"reviewed":"private"}'
            (root / 'compose.production.private.json').write_bytes(raw)
            receipt = {'root':str(root), 'phase': 'prepared', 'compose_sha256': promote.hashlib.sha256(raw).hexdigest(),
                       'before_write_fingerprint': 'before'}
            runner = Mock()
            with patch.object(promote.stack, 'checked_resume_root', return_value=root), \
                 patch.object(promote.audit, 'private_json', return_value=receipt), \
                 patch.object(promote, 'continuity', return_value='changed'), \
                 patch.object(promote.stack, 'Runner', return_value=runner):
                with self.assertRaises(promote.PromotionFailure): promote.activate(root)
            runner.run.assert_not_called()

    def test_cloud_rollback_after_writes_refused_and_local_api_restarted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt = {'root':str(root), 'phase': 'promoted', 'before_write_fingerprint': 'before'}
            runner = Mock()
            with patch.object(promote.stack, 'checked_resume_root', return_value=root), \
                 patch.object(promote.audit, 'private_json', return_value=receipt), \
                 patch.object(promote, 'continuity', return_value='new_owner_write'), \
                 patch.object(promote.stack, 'live_rows', return_value={}), \
                 patch.object(promote, 'reviewed_configs', return_value=({},{})), \
                 patch.object(promote, 'inspect_originals', return_value={}), \
                 patch.object(promote.stack, 'Runner', return_value=runner):
                with self.assertRaisesRegex(promote.PromotionFailure, 'Local writes exist'):
                    promote.rollback_before_writes(root)
            commands = [call.args[0] for call in runner.run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][-3:], ['stop', 'v3', 'api'])
            self.assertEqual(commands[1][-3:], ['--no-deps', 'v3', 'api'])
            self.assertFalse(any('rename' in command for command in commands))


class DurablePromotionTests(unittest.TestCase):
    """Stateful Docker fixtures fail after real simulated effects, then resume."""
    config = PromotionSafetyTests.config
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'selfhost-20261012T170000Z'
        self.root.mkdir()
        self.config_data = self.config()
        self.config_data['name'] = self.root.name.lower()
        self.source = {}
        for index,(service,name) in enumerate((('v3','arborscan-api'),('api','arborscan-api-v4'),('worker','arborscan-quality-worker'))):
            self.source[service]={'Id':'source-'+service,'Name':'/'+name,'Image':'sha256:'+'1'*64,
                'Config':{'Env':['SUPABASE_URL=https://cloud.invalid'],'Labels':{}},'Mounts':[],
                'HostConfig':{'PortBindings':{},'RestartPolicy':{'Name':'unless-stopped','MaximumRetryCount':0}},
                'State':{'Running':True}}
        self.rows = {row['Id']:copy.deepcopy(row) for row in self.source.values()}
        self.receipt={'format':1,'root':str(self.root),'phase':'prepared','before_write_fingerprint':'before','originals':[],
                      'release_gates': {'source_proof': str(self.root / 'source-proof.json'), 'source_proof_sha256': 'fixture-proof'}}
        for row in self.source.values():
            self.receipt['originals'].append({'id':row['Id'],'name':row['Name'][1:],
                'parked':row['Name'][1:]+'-cloud-rollback-'+self.root.name[9:],
                'restart':row['HostConfig']['RestartPolicy'],'running':True})
        candidate=promote.plan(self.config_data,self.source)
        promote.stack.private_write(self.root/'compose.private.json',json.dumps(self.config_data))
        promote.stack.private_write(self.root/'compose.production.private.json',json.dumps(candidate))
        self.receipt['compose_sha256']=promote.hashlib.sha256((self.root/'compose.production.private.json').read_bytes()).hexdigest()
        promote.stack.private_write(self.root/'promotion.private.json',json.dumps(self.receipt))
        self.actions=[];self.fail_after=None;self.fingerprint='before'
        self.addCleanup(patch.stopall)
        patch.object(promote.stack,'checked_resume_root',return_value=self.root).start()
        patch.object(promote.stack,'live_rows',return_value=self.source).start()
        patch.object(promote.stack,'Runner',return_value=SimpleNamespace(run=self.docker_run)).start()
        patch.object(promote,'continuity',side_effect=lambda root:self.fingerprint).start()
        patch.object(promote,'health').start()
        patch.object(promote.gates, 'source_gate', return_value={'source_proof_sha256': 'fixture-proof'}).start()

    def docker_run(self,command,data=None,required=True,timeout=None):
        if command[:2]==['docker','inspect']:
            return SimpleNamespace(stdout=json.dumps([self.rows[key] for key in command[2:]]).encode(),returncode=0)
        if command[:3]==['docker','ps','-aq']:
            name=command[-1][len('name=^'): -1]
            return SimpleNamespace(stdout='\n'.join(row['Id'] for row in self.rows.values() if row['Name']==name).encode(),returncode=0)
        receipt=json.loads((self.root/'promotion.private.json').read_text())
        self.actions.append(command)
        if command[1] in ('stop','update','rename','start'):
            identifier=command[-2] if command[1]=='rename' else command[-1]
            row=self.rows[identifier]
            self.assertIn(receipt['phase'],('parking_originals','rolling_back'))
            self.assertTrue((self.root/'compose.staging.private.json').is_file())
            if command[1]=='stop': row['State']['Running']=False
            elif command[1]=='update':
                row['HostConfig']['RestartPolicy']={'Name':command[-2].split('=',1)[1],'MaximumRetryCount':0}
            elif command[1]=='rename':
                self.assertFalse(any(other['Name']=='/'+command[-1] for other in self.rows.values() if other['Id']!=identifier))
                row['Name']='/'+command[-1]
            else:
                self.assertNotEqual(identifier,'source-worker')
                row['State']['Running']=True
            if self.fail_after==(command[1],identifier):
                self.fail_after=None
                raise RuntimeError('simulated interruption after Docker effect')
        elif command[1]=='compose':
            if 'stop' in command:
                for row in self.rows.values():
                    if row['Config']['Labels'].get('com.docker.compose.project')==self.config_data['name']:
                        row['State']['Running']=False
            else:
                cfg=json.loads((self.root/'compose.private.json').read_text())
                for service in ('v3','api'):
                    identifier='local-'+service
                    self.rows[identifier]={'Id':identifier,'Name':'/'+cfg['services'][service]['container_name'],
                        'Image':cfg['services'][service]['image'],'State':{'Running':True},
                        'Config':{'Labels':{'com.docker.compose.project':cfg['name'],'com.docker.compose.service':service}}}
                if self.fail_after == ('up', 'local'):
                    self.fail_after = None
                    raise RuntimeError('simulated interruption after local API start')
        return SimpleNamespace(stdout=b'',returncode=0)

    def test_failure_after_original_stop_resumes_using_id_and_durable_intent(self):
        self.fail_after=('stop','source-v3')
        with self.assertRaises(RuntimeError):promote.activate(self.root)
        receipt=json.loads((self.root/'promotion.private.json').read_text())
        self.assertEqual(receipt['phase'],'parking_originals')
        self.assertEqual(receipt['originals'][0]['step'],'stop_pending')
        result=promote.activate(self.root)
        self.assertEqual(result['promotion'],'passed')
        stops=[c for c in self.actions if c[1]=='stop' and c[-1]=='source-v3']
        self.assertEqual(len(stops),1)
        self.assertTrue(self.rows['local-v3']['State']['Running'])
        self.assertFalse(self.rows['source-worker']['State']['Running'])

    def test_failure_after_rename_can_rollback_partial_parking_and_resume_rollback(self):
        self.fail_after=('rename','source-api')
        with self.assertRaises(RuntimeError):promote.activate(self.root)
        self.fail_after=('rename','source-v3')
        with self.assertRaises(RuntimeError):promote.rollback_before_writes(self.root)
        self.assertEqual(json.loads((self.root/'promotion.private.json').read_text())['phase'],'rolling_back')
        result=promote.rollback_before_writes(self.root)
        self.assertEqual(result['rollback'],'original_cloud_api_restored')
        for service in ('v3','api'):
            self.assertEqual(self.rows['source-'+service]['Name'],self.source[service]['Name'])
            self.assertTrue(self.rows['source-'+service]['State']['Running'])
        self.assertFalse(self.rows['source-worker']['State']['Running'])
        self.assertFalse(any(c[1]=='start' and c[-1]=='source-worker' for c in self.actions))

    def test_promoted_restart_keeps_new_local_writes_and_cloud_rollback_refuses_them(self):
        promote.activate(self.root)
        self.fingerprint='new-local-report'
        self.assertEqual(promote.activate(self.root)['promotion'],'passed')
        before=len(self.actions)
        with self.assertRaisesRegex(promote.PromotionFailure,'Local writes exist'):
            promote.rollback_before_writes(self.root)
        actions=self.actions[before:]
        self.assertFalse(any(c[1] in ('rename','start','update') for c in actions))
        self.assertTrue(self.rows['local-v3']['State']['Running'])
        self.assertFalse(self.rows['source-v3']['State']['Running'])

    def test_full_rollback_parks_candidate_and_preserves_both_deployments(self):
        promote.activate(self.root)
        result=promote.rollback_before_writes(self.root)
        self.assertEqual(result['rollback'],'original_cloud_api_restored')
        self.assertFalse((self.root/'PROMOTED').exists())
        self.assertTrue((self.root/'PROMOTED.before-cloud-rollback').is_file())
        self.assertIn('-local-rollback-',self.rows['local-api']['Name'])
        self.assertFalse(self.rows['local-api']['State']['Running'])
        self.assertTrue(self.rows['source-api']['State']['Running'])
        self.assertEqual(len(self.rows),5)

    def test_resume_after_config_replace_keeps_saved_original_config(self):
        real=promote.atomic_private
        def interrupted(path,value):
            real(path,value)
            if path.name=='compose.private.json':raise RuntimeError('interrupted after atomic replacement')
        with patch.object(promote,'atomic_private',side_effect=interrupted):
            with self.assertRaises(RuntimeError):promote.activate(self.root)
        self.assertEqual(json.loads((self.root/'promotion.private.json').read_text())['phase'],'local_api_starting')
        self.assertEqual(json.loads((self.root/'compose.staging.private.json').read_text()),self.config_data)
        self.assertEqual(promote.activate(self.root)['promotion'],'passed')
        self.assertFalse(self.rows['source-worker']['State']['Running'])

    def test_resume_after_local_start_preserves_new_writes(self):
        self.fail_after=('up','local')
        with self.assertRaises(RuntimeError):promote.activate(self.root)
        self.assertEqual(json.loads((self.root/'promotion.private.json').read_text())['phase'],'local_api_starting')
        self.fingerprint='new-session-and-report'
        self.assertEqual(promote.activate(self.root)['promotion'],'passed')
        self.assertFalse(self.rows['source-v3']['State']['Running'])
        self.assertTrue(self.rows['local-v3']['State']['Running'])

    def test_rollback_resume_after_candidate_rename_uses_original_ids(self):
        promote.activate(self.root)
        self.fail_after=('rename','local-v3')
        with self.assertRaises(RuntimeError):promote.rollback_before_writes(self.root)
        self.assertEqual(json.loads((self.root/'promotion.private.json').read_text())['phase'],'rolling_back')
        self.assertEqual(promote.rollback_before_writes(self.root)['rollback'],'original_cloud_api_restored')
        self.assertEqual(self.rows['source-v3']['Name'],'/arborscan-api')
        self.assertIn('-local-rollback-',self.rows['local-v3']['Name'])

    def test_resume_refuses_unrelated_change_to_original_container(self):
        self.fail_after=('stop','source-v3')
        with self.assertRaises(RuntimeError):promote.activate(self.root)
        self.rows['source-api']['Config']['Env'].append('UNRELATED=change')
        before=len(self.actions)
        with self.assertRaisesRegex(promote.PromotionFailure,'Original deployment changed'):
            promote.activate(self.root)
        self.assertEqual(len(self.actions),before)

    def test_rollback_worker_restart_stays_disabled_while_api_policy_is_restored(self):
        for row in self.source.values():
            row['HostConfig']['RestartPolicy']['Name'] = 'always'
        for row in self.rows.values():
            row['HostConfig']['RestartPolicy']['Name'] = 'always'
        promote.atomic_private(self.root / 'promotion.private.json', self.receipt)
        promote.activate(self.root)
        promote.rollback_before_writes(self.root)
        self.assertEqual(self.rows['source-worker']['HostConfig']['RestartPolicy']['Name'], 'no')
        self.assertFalse(self.rows['source-worker']['State']['Running'])
        for service in ('v3', 'api'):
            self.assertEqual(self.rows['source-' + service]['HostConfig']['RestartPolicy']['Name'], 'always')
            self.assertTrue(self.rows['source-' + service]['State']['Running'])


if __name__ == '__main__': unittest.main()
