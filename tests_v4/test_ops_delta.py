import hashlib
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy-vps'))
from ops_delta_copy import BLOCK,reconstruct


class DeltaTest(unittest.TestCase):
 def test_exact_new_archive_not_mixed_versions(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'old').mkdir();(root/'new').mkdir()
   old=b'a'*BLOCK+b'b'*BLOCK+b'old tail';new=b'a'*BLOCK+b'c'*BLOCK+b'new tail'
   (root/'old/application.tar').write_bytes(old)
   blocks=[{'offset':i,'size':len(new[i:i+BLOCK]),'sha256':hashlib.sha256(new[i:i+BLOCK]).hexdigest()} for i in range(0,len(new),BLOCK)]
   record={'blocks':{'application.tar':blocks}};requests=[]
   def fetch(rows):requests.extend(rows);return b''.join(new[r['offset']:r['offset']+r['size']] for r in rows)
   target=root/'new/application.tar'
   self.assertTrue(reconstruct(root,record,'application.tar',target,fetch))
   self.assertEqual(target.read_bytes(),new);self.assertEqual(len(requests),2)
   self.assertEqual((root/'old/application.tar').read_bytes(),old)
 def test_corrupt_remote_block_not_published(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);target=root/'application.tar'
   record={'blocks':{'application.tar':[{'offset':0,'size':3,'sha256':hashlib.sha256(b'abc').hexdigest()}]}}
   with self.assertRaises(ValueError):reconstruct(root,record,'application.tar',target,lambda rows:b'bad')
   self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
