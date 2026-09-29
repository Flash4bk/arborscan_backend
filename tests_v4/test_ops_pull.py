import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

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


if __name__=='__main__':unittest.main()
