"""Tiny fixtures cover auth image loss, resumable copy and bounded retention."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]/'deploy-vps'))
import ops_backup_policy as policy
import ops_pull_backups as pull
import ops_windows_retention as retention

IMAGES = ['sha256:' + c*64 for c in '123']
SERVICES = sorted(retention.SERVICES)


class WindowsRuntimeRetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payloads = {f'/home/arborscan/runtime/{i}.tar': ('fixture image'+str(i)).encode()
                         for i in range(3)}

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name='20261008T140000Z', automatic=True, auth=True):
        folder = self.root/name
        folder.mkdir()
        for filename in policy.REQUIRED:
            (folder/filename).write_bytes(b'fixture '+filename.encode())
        services = SERVICES if auth else SERVICES[1:]
        rows = [{'Name':service,'Image':image} for service,image in zip(SERVICES,IMAGES)]
        (folder/'containers.private.json').write_text(json.dumps(rows))
        images = {row['Image']:[{'path':f'/home/arborscan/runtime/{i}.tar',
                    'sha256':hashlib.sha256(self.payloads[f'/home/arborscan/runtime/{i}.tar']).hexdigest()}]
                  for i,row in enumerate(rows) if row['Name'] in services}
        runtime = {'format':1,'images':images}
        if auth:
            runtime['services'] = services
        (folder/'RUNTIME_DEPENDENCIES.json').write_text(json.dumps(runtime))
        native = folder/'postgres'
        native.mkdir()
        for filename in policy.NATIVE:
            (native/filename).write_bytes(b'PGDMPfixture' if filename=='database.dump' else b'fixture')
        (native/'SHA256SUMS').write_text(''.join(retention.digest(native/n)+'  '+n+'\n'
                                              for n in sorted(policy.NATIVE)),newline='\n')
        (native/'COMPLETE').touch()
        if automatic:
            (folder/'AUTO_BACKUP.json').write_text(json.dumps({'format':1,'kind':'scheduled_backup','name':name}))
        policy.write_manifest(folder)
        (folder/'COMPLETE').touch()
        assets = [{'path':a['path'],'sha256':a['sha256'],'bytes':len(self.payloads[a['path']])}
                  for values in images.values() for a in values]
        return folder, {'name':name,'manifest':(folder/'SHA256SUMS').read_text(),
                        'runtime':runtime,'runtime_assets':assets,'bytes':100}

    def copy_runtime(self, folder, record, callback=None, free=None):
        def download_many(name, files):
            self.assertIsNone(name)
            for source,path in files:
                path.write_bytes(self.payloads[source])
        return pull.transfer_runtime(self.root, folder, record,
                callback or download_many, free or (lambda _:types.SimpleNamespace(free=10*1024**3)))

    def test_all_three_services_download_then_idempotent_readback(self):
        folder, record = self.fixture()
        result = self.copy_runtime(folder, record)
        self.assertEqual(result['runtime_services'],SERVICES)
        self.assertEqual(result['unique_runtime_archives'],3)
        self.copy_runtime(folder, record, lambda *_:self.fail('repeat downloaded runtime'))
        self.assertEqual(retention.verify_replacement(self.root,folder.name)['unique_runtime_archives'],3)

    def test_historical_dot_prefix_manifest_keeps_bytes_and_verifies(self):
        folder,record=self.fixture()
        checksum=folder/'SHA256SUMS'
        checksum.write_text(''.join(line.replace('  ','  ./',1)+'\n'
                                   for line in checksum.read_text().splitlines()),newline='\n')
        original=checksum.read_bytes()
        entries=retention.verify_data(folder)
        self.assertIn('application.tar',entries)
        self.assertEqual(checksum.read_bytes(),original)
        record['manifest']=checksum.read_text()
        self.copy_runtime(folder,record)
        retention.verify_replacement(self.root,folder.name)

    def test_canonical_duplicate_and_traversal_manifest_refused(self):
        folder,_=self.fixture()
        checksum=folder/'SHA256SUMS'
        original=checksum.read_text()
        checksum.write_text(original+retention.digest(folder/'application.tar')+'  ./application.tar\n')
        with self.assertRaisesRegex(ValueError,'invalid_manifest'):
            retention.verify_data(folder)
        checksum.write_text(original+'0'*64+'  ../outside\n')
        with self.assertRaisesRegex(ValueError,'invalid_manifest'):
            retention.verify_data(folder)

    def test_interrupted_runtime_keeps_partial_and_retry_publishes(self):
        folder,record = self.fixture()
        (folder/'COMPLETE').unlink()
        def interrupted(_,files):
            source,path = files[0]
            path.write_bytes(self.payloads[source][:3])
            raise RuntimeError('transfer_interrupted')
        with self.assertRaises(RuntimeError):
            self.copy_runtime(folder,record,interrupted)
        self.assertFalse((folder/'COMPLETE').exists())
        self.assertFalse((folder/'RELOCATED_RUNTIME.json').exists())
        self.copy_runtime(folder,record)
        self.assertEqual(len(list((self.root/'runtime-assets').glob('*.archive'))),3)

    def test_bad_runtime_sha_cannot_authorize_retention(self):
        folder,record=self.fixture()
        def damaged(_,files):
            for _,path in files:path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum_mismatch'):
            self.copy_runtime(folder,record,damaged)
        with self.assertRaises((OSError,ValueError)):
            retention.plan(self.root,folder.name)
        self.assertTrue(folder.exists())

    def test_runtime_disk_guard_before_download_or_retention(self):
        folder,record=self.fixture()
        with self.assertRaisesRegex(ValueError,'insufficient_disk'):
            self.copy_runtime(folder,record,lambda *_:self.fail('low disk downloaded'),
                              lambda _:types.SimpleNamespace(free=1))
        self.assertTrue(folder.exists())

    def test_shared_runtime_reused_between_sets(self):
        first,record=self.fixture('20261008T130000Z')
        self.copy_runtime(first,record)
        second,record=self.fixture()
        self.copy_runtime(second,record,lambda *_:self.fail('shared bytes duplicated'))
        shutil.rmtree(first)
        retention.verify_replacement(self.root,second.name)

    def test_missing_or_changed_runtime_contract_refused(self):
        folder,record=self.fixture()
        record['runtime_assets'].pop()
        with self.assertRaisesRegex(ValueError,'runtime_asset_inventory_mismatch'):
            self.copy_runtime(folder,record)

    def test_new_snapshot_refuses_missing_auth_image_in_index(self):
        folder,record=self.fixture()
        assets=self.root/'fixture-runtime.tar';assets.write_bytes(b'runtime')
        index=self.root/'index.json'
        index.write_text(json.dumps({'format':1,'images':{i:[{'path':str(assets),
                    'sha256':retention.digest(assets)}] for i in IMAGES[1:]}}))
        target=self.root/'new-root';target.mkdir()
        def create(attempt):
            for f in folder.iterdir():
                if f.is_file() and f.name not in ('COMPLETE','RUNTIME_DEPENDENCIES.json','AUTO_BACKUP.json'):
                    shutil.copyfile(f,attempt/f.name)
            shutil.copytree(folder/'postgres',attempt/'postgres')
        with self.assertRaisesRegex(policy.PolicyError,'runtime_archive_index_missing'):
            policy.run_backup(target,create,index,operation_id='new-auth',minimum_free=1)
        self.assertFalse(any((p/'COMPLETE').exists() for p in target.iterdir() if p.is_dir()))

    def test_new_snapshot_publishes_hash_covered_all_three_services(self):
        folder,record=self.fixture()
        asset=self.root/'complete-runtime.tar';asset.write_bytes(b'complete runtime')
        index=self.root/'index.json'
        index.write_text(json.dumps({'format':1,'images':{i:[{'path':str(asset),
                    'sha256':retention.digest(asset)}] for i in IMAGES}}))
        target=self.root/'new-root';target.mkdir()
        def create(attempt):
            for f in folder.iterdir():
                if f.is_file() and f.name not in ('COMPLETE','RUNTIME_DEPENDENCIES.json','AUTO_BACKUP.json'):
                    shutil.copyfile(f,attempt/f.name)
            shutil.copytree(folder/'postgres',attempt/'postgres')
        result=policy.run_backup(target,create,index,operation_id='new-auth',minimum_free=1)
        checked=policy.verify_set(target/result['name'],require_native=True,require_runtime=True)
        self.assertEqual(checked['runtime_services'],SERVICES)
        self.assertIn('RUNTIME_DEPENDENCIES.json',policy.manifest(target/result['name']))

    def test_legacy_runtime_scope_preserved_but_not_current_replacement(self):
        folder,record=self.fixture(auth=False)
        result=self.copy_runtime(folder,record)
        self.assertEqual(len(result['runtime_services']),2)
        with self.assertRaisesRegex(ValueError,'auth_runtime_required'):
            retention.verify_replacement(self.root,folder.name)

    def test_verified_new_replacement_then_rotation14_preserves_pin_and_shared(self):
        folders=[]
        for i in range(15):
            folder,record=self.fixture('20261008T%06dZ'%i)
            folders.append(folder)
        (folders[0]/'PINNED').touch()
        latest,record=self.fixture()
        self.copy_runtime(latest,record)
        proposal=retention.plan(self.root,latest.name)
        self.assertEqual(proposal['delete'],[folders[1].name,folders[2].name])
        result=retention.apply(self.root,proposal)
        self.assertEqual(result['retained_verified_sets'],14)
        self.assertTrue(folders[0].exists())
        self.assertTrue(latest.exists())
        self.assertEqual(len(list((self.root/'runtime-assets').glob('*.archive'))),3)
        self.assertEqual(retention.plan(self.root,latest.name)['delete'],[])

    def test_pull_dry_run_verifies_and_writes_plan_without_deleting(self):
        for i in range(14):self.fixture('20261008T%06dZ'%i)
        latest,record=self.fixture();self.copy_runtime(latest,record)
        locking=types.SimpleNamespace(LK_NBLCK=1,locking=lambda *_:None)
        job=types.SimpleNamespace(contain_children=lambda:None)
        with patch('sys.argv',['pull','--root',str(self.root),'--retention-dry-run']), \
                patch.dict(sys.modules,{'msvcrt':locking,'ops_windows_job':job}), \
                patch.dict(os.environ,{'WINDIR':r'C:\Windows'}), \
                patch.object(subprocess,'CREATE_NO_WINDOW',0,create=True), \
                patch.object(pull,'read_inventory',return_value=[record]), \
                patch.object(pull,'transfer',return_value='already_verified'), \
                patch.object(retention,'apply',side_effect=AssertionError('dry run deleted')):
            self.assertEqual(pull.main(),0)
        proposal=json.loads((self.root/'retention-dry-run.private.json').read_text())
        self.assertEqual(proposal['before'],15)
        self.assertEqual(len(proposal['delete']),1)
        self.assertEqual(len([p for p in self.root.iterdir() if (p/'COMPLETE').exists()]),15)

    def test_unadopted_history_protected_and_wrong_sha_cannot_adopt(self):
        for i in range(14):self.fixture('20261008T%06dZ'%i,automatic=False)
        latest,record=self.fixture();self.copy_runtime(latest,record)
        (self.root/'WINDOWS_RETENTION_POLICY.private.json').write_text(json.dumps({
            'format':1,'retention':14,'automatic_sets':[{'name':'20261008T000000Z','manifest_sha256':'0'*64}]}))
        config=json.loads((self.root/'WINDOWS_RETENTION_POLICY.private.json').read_text())
        config['automatic_sets'][0].update(source='observed_automatic_run',evidence_reference='fixture observed service')
        (self.root/'WINDOWS_RETENTION_POLICY.private.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'rotation_blocked_all_remaining_sets_protected'):
            retention.plan(self.root,latest.name)
        self.assertEqual(len([p for p in self.root.iterdir() if (p/'COMPLETE').is_file()]),15)

    def test_changed_replacement_refuses_all_deletion(self):
        for i in range(14):self.fixture('20261008T%06dZ'%i)
        latest,record=self.fixture();self.copy_runtime(latest,record)
        proposal=retention.plan(self.root,latest.name)
        (latest/'postgres/database.dump').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum_mismatch'):
            retention.apply(self.root,proposal)
        self.assertTrue((self.root/'20261008T000000Z').exists())

    def test_live_dependency_and_newer_sets_protected(self):
        folders=[]
        for i in range(14):folders.append(self.fixture('20261008T%06dZ'%i)[0])
        latest,record=self.fixture();self.copy_runtime(latest,record)
        proposal=retention.plan(self.root,latest.name,protected=[folders[0].name])
        self.assertEqual(proposal['delete'],[folders[1].name])

    def test_runtime_map_path_escape_refused(self):
        folder,record=self.fixture();self.copy_runtime(folder,record)
        mapping=json.loads((folder/'RELOCATED_RUNTIME.json').read_text())
        mapping['assets'][0]['restored_path']=str(self.root.parent/'outside.archive')
        (folder/'RELOCATED_RUNTIME.json').write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError,'unsafe_destination'):
            retention.verify_replacement(self.root,folder.name)

    def test_concurrent_retention_refused(self):
        latest,record=self.fixture();self.copy_runtime(latest,record)
        proposal=retention.plan(self.root,latest.name)
        with retention.locked(self.root),self.assertRaisesRegex(ValueError,'retention_already_running'):
            retention.apply(self.root,proposal)

    def test_unreadable_retained_dependencies_block_rotation(self):
        old,_=self.fixture('20261008T130000Z')
        latest,record=self.fixture();self.copy_runtime(latest,record)
        (old/'containers.private.json').write_bytes(b'corrupt metadata')
        with self.assertRaisesRegex(ValueError,'dependency_inventory_unreadable'):
            retention.plan(self.root,latest.name)

    @unittest.skipUnless(os.name=='nt','Windows junction guard')
    def test_actual_windows_junction_parent_refused(self):
        inside=self.root/'inside';inside.mkdir()
        junction=self.root/'junction'
        command="New-Item -ItemType Junction -Path '"+str(junction).replace("'","''")+"' -Target '"+str(inside).replace("'","''")+"' | Out-Null"
        subprocess.run(['powershell','-NoProfile','-Command',command],check=True,capture_output=True)
        with self.assertRaisesRegex(ValueError,'symlink_or_reparse_refused'):
            retention.safe_path(self.root,junction/'child')

    @unittest.skipUnless(os.name=='nt','Windows root junction guard')
    def test_actual_windows_root_junction_refused_before_resolve(self):
        latest,record=self.fixture();self.copy_runtime(latest,record)
        proposal=retention.plan(self.root,latest.name)
        junction=self.root/'alias-root'
        command="New-Item -ItemType Junction -Path '"+str(junction).replace("'","''")+"' -Target '"+str(self.root).replace("'","''")+"' | Out-Null"
        subprocess.run(['powershell','-NoProfile','-Command',command],check=True,capture_output=True)
        try:
            with self.assertRaisesRegex(ValueError,'symlink_or_reparse_refused'):
                retention.plan(junction,latest.name)
            with self.assertRaisesRegex(ValueError,'symlink_or_reparse_refused'):
                retention.apply(junction,proposal)
            with patch('sys.argv',['pull','--root',str(junction)]),self.assertRaisesRegex(ValueError,'symlink_or_reparse_refused'):
                pull.main()
        finally:
            junction.rmdir()  # Remove only the junction, never its target.


if __name__=='__main__':unittest.main()
