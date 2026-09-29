"""Read verified ranges of completed backup files over authenticated SSH only."""
import gzip,hashlib,json,re,sys
from pathlib import Path
root=Path('/home/arborscan/ops-backups')
request=json.load(sys.stdin)
name=request['backup'];filename=request['file']
assert re.fullmatch(r'\d{8}T\d{6}Z',name)
folder=root/name;path=(folder/filename).resolve()
assert path.is_relative_to(folder) and (folder/'COMPLETE').is_file()
manifest={str(Path(n)):h for h,n in (line.split('  ',1) for line in (folder/'SHA256SUMS').read_text().splitlines())}
assert manifest[filename]==request['sha256']
chunks=request['chunks'];assert sum(c['size'] for c in chunks)<=32*1024**2
with path.open('rb') as source,gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as output:
 for chunk in chunks:
  assert 0<=chunk['offset'] and 0<chunk['size']<=4*1024**2
  source.seek(chunk['offset']);raw=source.read(chunk['size'])
  assert len(raw)==chunk['size'] and hashlib.sha256(raw).hexdigest()==chunk['sha256']
  output.write(raw)
