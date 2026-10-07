import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy-vps'))
from ops_pull_backups import entries, transfer, read_inventory, INVENTORY


class PullTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.files={n:b'fixture-'+n.encode() for n in
                    ('application.tar','local-files.tar','arborscan.env.private','containers.private.json',
                     'postgres/database.dump','postgres/roles.sql','postgres/source.json','postgres/archive-list.private.txt','postgres/COMPLETE')}
        self.record={'name':'20260928T133059Z','bytes':1000,
                     'manifest':''.join(hashlib.sha256(b).hexdigest()+'  '+n+'\n' for n,b in self.files.items())}
    def tearDown(self):self.temp.cleanup()
    def download(self,name,file,path):path.write_bytes(self.files[file])
    def test_interrupted_resume_and_repeat(self):
        count=0
        def interrupted(name,file,path):
            nonlocal count
            count+=1
            if count==3:
                path.write_bytes(b'partial');raise OSError('test disconnect')
            self.download(name,file,path)
        with self.assertRaises(OSError):transfer(self.root,self.record,interrupted)
        self.assertFalse((self.root/self.record['name']).exists())
        downloaded=[]
        def resume(name,file,path):downloaded.append(file);self.download(name,file,path)
        self.assertEqual(transfer(self.root,self.record,resume),'downloaded_verified')
        self.assertNotIn('application.tar',downloaded)
        self.assertEqual(transfer(self.root,self.record,lambda *a:self.fail('duplicate download')),'verified_existing')
        self.assertTrue((self.root/self.record['name']/'COMPLETE').is_file())
    def test_bad_hash_never_published(self):
        with self.assertRaises(ValueError):transfer(self.root,self.record,lambda n,f,p:p.write_bytes(b'bad'))
        self.assertFalse((self.root/self.record['name']).exists())
    def test_unsafe_manifest(self):
        for name in ('../outside','/outside','C:/secret','a;whoami','a\\b'):
            with self.assertRaises(ValueError):entries('a'*64+'  '+name)
    def test_low_disk(self):
        with self.assertRaises(ValueError):transfer(self.root,self.record,self.download,lambda p:shutil._ntuple_diskusage(1,1,0))
        self.assertFalse((self.root/self.record['name']).exists())
    def test_covered_native_checksum_preserved_exactly_on_windows(self):
        native=('database.dump','roles.sql','source.json','archive-list.private.txt')
        original=''.join(hashlib.sha256(self.files['postgres/'+n]).hexdigest()+'  '+n+'\n' for n in native).encode()
        self.files['postgres/SHA256SUMS']=original
        self.record['manifest']=''.join(hashlib.sha256(b).hexdigest()+'  '+n+'\n' for n,b in self.files.items())
        transfer(self.root,self.record,self.download)
        destination=self.root/self.record['name']
        self.assertEqual((destination/'postgres/SHA256SUMS').read_bytes(),original)
        from ops_verify_offsite import verify
        def check_existing(path):
            self.assertTrue((path/'COMPLETE').exists())
            return verify(path)
        with patch('ops_pull_backups.verify',side_effect=check_existing):
            self.assertEqual(transfer(self.root,self.record,self.download),'verified_existing')
        self.assertEqual((destination/'postgres/SHA256SUMS').read_bytes(),original)
    def test_old_native_checksum_reconstructed_with_lf_bytes(self):
        transfer(self.root,self.record,self.download)
        raw=(self.root/self.record['name']/'postgres/SHA256SUMS').read_bytes()
        self.assertNotIn(b'\r\n',raw)
        self.assertEqual(len(raw.splitlines()),4)
    def test_outer_manifest_preserves_trusted_source_bytes_on_windows(self):
        transfer(self.root,self.record,self.download)
        expected=self.record['manifest'].encode('utf-8')
        manifest=self.root/self.record['name']/'SHA256SUMS'
        self.assertEqual(manifest.read_bytes(),expected)
        self.assertEqual(transfer(self.root,self.record,lambda *a:self.fail('duplicate download')),'verified_existing')
        self.assertEqual(manifest.read_bytes(),expected)


class InventoryRetryTest(unittest.TestCase):
    def setUp(self):
        self.command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','private-host','python3 -']
        self.record=[{'name':'20261006T195518Z','manifest':'fixture','bytes':42}]
        self.calls=[]; self.logs=[]; self.waits=[]

    def runner(self, replies):
        iterator=iter(replies)
        def run(command, **kwargs):
            self.calls.append((command,kwargs))
            reply=next(iterator)
            if isinstance(reply,BaseException):raise reply
            return reply
        return run

    def log(self,event,**fields):self.logs.append({'event':event,**fields})
    def success(self):return subprocess.CompletedProcess(self.command,0,json.dumps(self.record).encode(),b'')
    def failed(self):return subprocess.CompletedProcess(self.command,255,b'private stdout',b'Connection closed; secret=PRIVATE_SENTINEL /private/location')

    def read(self,replies):
        return read_inventory(self.command,hidden={'creationflags':123},run=self.runner(replies),
                              sleep=self.waits.append,log=self.log)

    def test_transient_connection_error_recovers_without_changing_authentication(self):
        self.assertEqual(self.read([self.failed(),self.success()]),self.record)
        self.assertEqual(len(self.calls),2)
        for command,kwargs in self.calls:
            self.assertEqual(command,self.command)
            self.assertEqual(kwargs,{'input':INVENTORY.encode(),'capture_output':True,'timeout':180,'creationflags':123})
        self.assertEqual(self.waits,[5])
        self.assertEqual(self.logs,[{'event':'inventory_attempt_failed','attempt':1,
                                   'exit_code':255,'categories':['connection_closed']}])

    def test_exhaustion_is_three_attempts_and_no_untrusted_diagnostics(self):
        with self.assertRaisesRegex(RuntimeError,'^ssh_inventory_failed$'):
            self.read([self.failed()]*4)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(self.waits,[5,10])
        self.assertEqual([r['attempt'] for r in self.logs],[1,2,3])
        self.assertNotIn('PRIVATE_SENTINEL',json.dumps(self.logs))
        self.assertNotIn('/private/location',json.dumps(self.logs))
        self.assertNotIn('private stdout',json.dumps(self.logs))

    def test_timeout_is_retried_and_timeout_output_is_redacted(self):
        timeout=subprocess.TimeoutExpired(self.command,180,output=b'PRIVATE_SENTINEL',stderr=b'/private/location')
        self.assertEqual(self.read([timeout,self.success()]),self.record)
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.logs,[{'event':'inventory_attempt_failed','attempt':1,'categories':['timeout']}])
        self.assertNotIn('PRIVATE_SENTINEL',json.dumps(self.logs))

    def test_success_is_not_repeated(self):
        self.assertEqual(self.read([self.success()]),self.record)
        self.assertEqual(len(self.calls),1)
        self.assertFalse(self.waits)
        self.assertFalse(self.logs)

    def test_invalid_or_empty_response_fails_without_transport_retry(self):
        for raw in (b'PRIVATE_SENTINEL',b'{}',b'[]'):
            with self.subTest(response=raw):
                self.calls.clear();self.logs.clear();self.waits.clear()
                reply=subprocess.CompletedProcess(self.command,0,raw,b'')
                with self.assertRaisesRegex(RuntimeError,'^ssh_inventory_invalid_response$'):
                    self.read([reply,self.success()])
                self.assertEqual(len(self.calls),1)
                self.assertFalse(self.waits)

    def test_missing_local_client_fails_safely_without_retry(self):
        with self.assertRaisesRegex(RuntimeError,'^ssh_inventory_failed$'):
            self.read([FileNotFoundError('PRIVATE_SENTINEL /private/location'),self.success()])
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.logs,[{'event':'inventory_attempt_failed','attempt':1,'categories':['local_client_error']}])

    def test_authentication_and_host_key_failures_never_change_secure_arguments(self):
        for stderr,category in ((b'Permission denied (publickey); PRIVATE_SENTINEL','authentication_denied'),
                                (b'Host key verification failed /private/location','host_key_rejected')):
            with self.subTest(category=category):
                self.calls.clear();self.logs.clear();self.waits.clear()
                reply=subprocess.CompletedProcess(self.command,255,b'',stderr)
                with self.assertRaisesRegex(RuntimeError,'^ssh_inventory_failed$'):
                    self.read([reply]*3)
                self.assertEqual(len(self.calls),3)
                self.assertTrue(all(command==self.command for command,_ in self.calls))
                self.assertTrue(all(row['categories']==[category] for row in self.logs))
                self.assertNotIn('PRIVATE_SENTINEL',json.dumps(self.logs))
                self.assertNotIn('/private/location',json.dumps(self.logs))


if __name__=='__main__':unittest.main()
