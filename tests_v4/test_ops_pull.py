import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy-vps'))
from ops_pull_backups import entries, transfer


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


if __name__=='__main__':unittest.main()
