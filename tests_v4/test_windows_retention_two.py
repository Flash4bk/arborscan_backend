"""Explicit local keep-two policy, legacy default and real Windows deletion bounds."""
import hashlib
import importlib.util
import io
import base64
import json
import os
from pathlib import Path
import sys
import tarfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'deploy-vps'))
import ops_local_retention as compact
import ops_pull_backups as pull
import ops_windows_retention as retention
import ops_verify_restore as restore_module

spec=importlib.util.spec_from_file_location('runtime_fixtures',Path(__file__).with_name('test_ops_windows_runtime_retention.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
Base=fixtures.WindowsRuntimeRetentionTest


class LocalLatestTwoTest(unittest.TestCase):
    setUp=Base.setUp
    tearDown=Base.tearDown
    fixture=Base.fixture
    copy_runtime=Base.copy_runtime

    def configured(self,count=4,manual=False):
        rows=[]
        for i in range(count):
            folder,record=self.fixture('20261008T%06dZ'%i,automatic=not manual)
            self.copy_runtime(folder,record)
            rows.append((folder,record))
        value={'format':1,'kind':'owner_selected_local_latest_two','retain':2,
               'authorization':'explicit_owner_request','root':str(self.root.resolve()),
               'authorized_at':'2026-10-31T12:00:00Z',
               'released_sets':[{'name':p.name,'manifest_sha256':retention.digest(p/'SHA256SUMS'),'pins':[]}
                                for p,_ in rows[:-2]]}
        (self.root/compact.POLICY_FILE).write_text(json.dumps(value),encoding='utf-8')
        return rows,value

    def test_missing_policy_preserves_default_record_list(self):
        records=[{'not':'a backup'}]
        self.assertIsNone(compact.load_policy(self.root))
        self.assertIs(compact.select_records(records,None),records)

    def test_policy_cannot_silently_target_other_root_or_retention(self):
        _,value=self.configured()
        for key,new in [('root',str(self.root.parent)),('retain',1),('authorization','automatic')]:
            changed={**value,key:new}
            (self.root/compact.POLICY_FILE).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'invalid_local_latest_two_policy'):
                compact.load_policy(self.root)

    def test_source_selection_precedes_transfer_and_keeps_only_latest_two(self):
        rows,value=self.configured(5)
        selected=compact.select_records([r for _,r in reversed(rows)],value)
        self.assertEqual([r['name'] for r in selected],[p.name for p,_ in rows[-2:]])

    @unittest.skipUnless(os.name=='nt','Windows pull entry point')
    def test_reader_dry_run_selects_before_transfer_and_default14_is_unchanged(self):
        rows,_=self.configured(4)
        records=[r for _,r in rows]
        for compact_mode in [True,False]:
            if not compact_mode:(self.root/compact.POLICY_FILE).unlink()
            with patch.object(pull,'read_inventory',return_value=records), \
                    patch.object(pull,'transfer',return_value='existing_verified') as transfer, \
                    patch.dict(sys.modules,{'ops_windows_job':types.SimpleNamespace(contain_children=lambda:None)}), \
                    patch('sys.argv',['pull','--root',str(self.root),'--retention-dry-run']):
                self.assertEqual(pull.main(),0)
            self.assertEqual([call.args[1]['name'] for call in transfer.call_args_list],
                             [p.name for p,_ in (rows[-2:] if compact_mode else rows)])
            self.assertEqual(len([p for p,_ in rows if p.exists()]),4)

    def test_installed_reader_adapter_preserves_existing_fixes_and_is_idempotent(self):
        source='''import os
def main():
    try:
            records=read_inventory(known=known_local_manifests(root),timeout=args.inventory_timeout)
            def delta(record,filename,target):
                return existing_resume(record)
            preserve_transition_history(root, records)
            latest = records[-1]
    except Exception:
            return 1
def existing_selfhost_check():
    return preserve_all_seven_roles()
'''
        changed=compact.adapt_reader_source(source)
        compile(changed,'fixture_reader.py','exec')
        self.assertIn('known=known_local_manifests(root),timeout=args.inventory_timeout',changed)
        self.assertIn('return existing_resume(record)',changed)
        self.assertIn('return preserve_all_seven_roles()',changed)
        self.assertIn('if compact_policy is None:\n                preserve_transition_history',changed)
        self.assertLess(changed.index('records = select_records'),changed.index('def delta'))
        self.assertEqual(compact.adapt_reader_source(changed),changed)

    def test_duplicate_or_insufficient_remote_inventory_refused(self):
        rows,value=self.configured(2)
        with self.assertRaisesRegex(ValueError,'invalid_local_source_inventory'):
            compact.select_records([rows[0][1],rows[0][1]],value)
        with self.assertRaisesRegex(ValueError,'two_native_runtime_backups_required'):
            compact.select_records([rows[0][1]],value)

    def test_explicit_release_covers_old_manual_history(self):
        rows,_=self.configured(manual=True)
        proposal=compact.plan(self.root)
        self.assertEqual([r['name'] for r in proposal['delete']],[p.name for p,_ in rows[:-2]])
        self.assertEqual(len(proposal['shared_runtime']),3)

    def test_changed_release_manifest_or_pin_blocks_before_deletion(self):
        rows,value=self.configured()
        (rows[0][0]/'PINNED').touch()
        with self.assertRaisesRegex(ValueError,'released_backup_changed'):
            compact.plan(self.root)
        (rows[0][0]/'PINNED').unlink()
        (rows[0][0]/'SHA256SUMS').write_text('changed')
        with self.assertRaisesRegex(ValueError,'released_backup_changed'):
            compact.plan(self.root)

    def test_changed_kept_postgres_or_shared_runtime_blocks(self):
        rows,_=self.configured()
        (rows[-1][0]/'postgres/database.dump').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum_mismatch'):
            compact.plan(self.root)

    def test_local_runtime_dependency_cannot_be_deleted(self):
        rows,_=self.configured()
        # A valid checksum relocated into old dated history is still a dependency.
        old=rows[0][0];latest=rows[-1][0]
        mapping=json.loads((latest/'RELOCATED_RUNTIME.json').read_text())
        source=Path(mapping['assets'][0]['restored_path'])
        dependency=old/'saved-image.archive';dependency.write_bytes(source.read_bytes())
        mapping['assets'][0]['restored_path']=str(dependency)
        (latest/'RELOCATED_RUNTIME.json').write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError,'kept_backup_depends_on_disposable_directory'):
            compact.plan(self.root)

    def test_stale_source_cannot_delete_newer_local_backup(self):
        rows,_=self.configured()
        with self.assertRaisesRegex(ValueError,'local_source_inventory_is_stale'):
            compact.plan(self.root,selected=[rows[0][0].name,rows[1][0].name])

    def test_future_automatic_set_rotates_previous_two_without_resurrection(self):
        rows,value=self.configured(2)
        p,r=self.fixture('20261009T000000Z');self.copy_runtime(p,r)
        proposal=compact.plan(self.root)
        self.assertEqual([x['name'] for x in proposal['delete']],[rows[0][0].name])
        selected=compact.select_records([x[1] for x in rows]+[r],value)
        self.assertEqual([x['name'] for x in selected],[rows[1][0].name,p.name])

    def test_unknown_manual_or_new_pin_is_not_released(self):
        rows,_=self.configured(2)
        old,r=self.fixture('20261007T000000Z',automatic=False);self.copy_runtime(old,r)
        with self.assertRaisesRegex(ValueError,'unreleased_historical_backup'):
            compact.plan(self.root)
        (old/'PINNED').touch()
        with self.assertRaisesRegex(ValueError,'new_pinned_backup_is_not_released'):
            compact.plan(self.root)

    def test_plan_change_prevents_native_delete(self):
        rows,_=self.configured()
        proposal=compact.plan(self.root)
        policy=self.root/compact.POLICY_FILE
        policy.write_text(policy.read_text()+'\n')
        with patch.object(compact,'_powershell_remove') as remove:
            with self.assertRaisesRegex(ValueError,'local_retention_plan_changed'):
                compact.apply(self.root,proposal)
            remove.assert_not_called()

    def test_prepared_policy_produces_real_readonly_plan_and_cannot_delete(self):
        rows,_=self.configured()
        active=self.root/compact.POLICY_FILE
        active.rename(self.root/compact.PREPARED_POLICY_FILE)
        self.assertIsNone(compact.load_policy(self.root))
        proposal=compact.plan(self.root,policy_file=compact.PREPARED_POLICY_FILE)
        self.assertTrue(proposal['prepared_only'])
        self.assertEqual(len(proposal['delete']),2)
        with patch.object(compact,'_powershell_remove') as remove:
            with self.assertRaisesRegex(ValueError,'prepared_local_policy_cannot_delete'):
                compact.apply(self.root,proposal)
            remove.assert_not_called()
        self.assertTrue(all(p.exists() for p,_ in rows))

    @unittest.skipUnless(os.name=='nt','Native Windows removal and byte lock')
    def test_real_native_removal_only_fixture_sets_preserves_shared_tools_and_key(self):
        rows,_=self.configured()
        tools=self.root/'tools';tools.mkdir();(tools/'fixture.py').write_bytes(b'tool')
        (self.root/'signing-fixture.key').write_bytes(b'test-key-not-secret')
        result=compact.apply(self.root,compact.plan(self.root))
        self.assertEqual(result['retained_verified_sets'],2)
        self.assertEqual([p.name for p,_ in rows if p.exists()],[p.name for p,_ in rows[-2:]])
        self.assertTrue((tools/'fixture.py').exists())
        self.assertEqual((self.root/'signing-fixture.key').read_bytes(),b'test-key-not-secret')
        self.assertEqual(len(list((self.root/'runtime-assets').glob('*.archive'))),3)

    def _inject_before_native_remove(self,script_line):
        original=compact.subprocess.run
        def change(arguments,**kwargs):
            arguments=list(arguments)
            script=base64.b64decode(arguments[-1]).decode('utf-16-le')
            marker='foreach($row in $data.rows) {\n  Assert-Root'
            self.assertEqual(script.count(marker),1)
            script=script.replace(marker,script_line+'\n'+marker)
            arguments[-1]=base64.b64encode(script.encode('utf-16-le')).decode()
            return original(arguments,**kwargs)
        return patch.object(compact.subprocess,'run',change)

    @unittest.skipUnless(os.name=='nt','Native Windows pin race')
    def test_pin_added_after_native_validation_blocks_removal(self):
        rows,_=self.configured()
        insertion="[IO.File]::WriteAllText((Join-Path $root '"+rows[0][0].name+"/PINNED'),'')"
        with self._inject_before_native_remove(insertion):
            with self.assertRaisesRegex(ValueError,'native_windows_removal_failed'):
                compact.apply(self.root,compact.plan(self.root))
        self.assertTrue(all(p.exists() for p,_ in rows))

    @unittest.skipUnless(os.name=='nt','Native Windows policy race')
    def test_policy_changed_after_native_validation_blocks_removal(self):
        rows,_=self.configured()
        insertion="[IO.File]::AppendAllText((Join-Path $root $data.policy_file),' ')"
        with self._inject_before_native_remove(insertion):
            with self.assertRaisesRegex(ValueError,'native_windows_removal_failed'):
                compact.apply(self.root,compact.plan(self.root))
        self.assertTrue(all(p.exists() for p,_ in rows))

    @unittest.skipUnless(os.name=='nt','Installed Windows byte lock')
    def test_concurrent_pull_lock_blocks_compact_apply(self):
        self.configured()
        proposal=compact.plan(self.root)
        with retention.locked(self.root):
            with self.assertRaisesRegex(ValueError,'retention_already_running'):
                compact.apply(self.root,proposal)


class RestoreUtf8Test(unittest.TestCase):
    def test_windows_drive_paths_and_ads_are_refused(self):
        import tempfile
        for name in ['C:/escape.txt','C:escape.txt','safe.txt:stream','../escape.txt']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                base=Path(temp);archive=base/'bad.tar'
                files={name:b'fixture', 'manifest.json':json.dumps({name:hashlib.sha256(b'fixture').hexdigest()}).encode()}
                with tarfile.open(archive,'w') as tar:
                    for filename,raw in files.items():
                        info=tarfile.TarInfo(filename);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
                with self.assertRaisesRegex(ValueError,'Unsafe archive'):
                    restore_module.restore(archive,base/'restored')
                self.assertEqual(list((base/'restored').iterdir()),[])

    def test_utf8_report_versions_survive_non_utf8_windows_default(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);payload='{"вид":"Осина"}'.encode('utf-8')
            digest=hashlib.sha256(payload).hexdigest()
            files={'tables/report_versions.json':json.dumps([{'owner_id':'fixture',
                    'payload_sha256':digest,'name':'Отчёт'}],ensure_ascii=False).encode('utf-8'),
                   'objects/private/report-versions/fixture/'+digest+'.json':payload}
            files['manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest()
                                              for k,v in files.items()}).encode()
            archive=base/'application.tar'
            with tarfile.open(archive,'w') as tar:
                for name,raw in files.items():
                    info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
            original=Path.read_text
            def non_utf8_default(path,*args,**kwargs):
                if path.name=='report_versions.json':
                    self.assertEqual(kwargs.get('encoding'),'utf-8')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',non_utf8_default):
                result=restore_module.restore(archive,base/'restored')
            self.assertEqual(result['reports_linked'],1)
            self.assertFalse(result['database_restored'])


del Base  # Fixture methods are reused; do not rediscover its full test class.
if __name__=='__main__':unittest.main()
