"""Import Storage bytes into the isolated local recovery service only."""
import hashlib,json,mimetypes,sys,time,urllib.request,urllib.parse
from pathlib import Path

root=Path(sys.argv[1]).resolve()
assert root.parent==Path('/home/arborscan') and root.name.startswith('as14-recovery-')
cfg=json.loads((root/'compose.private.json').read_text())
assert cfg['networks']['default']['internal'] is True
assert cfg['services']['gateway']['ports']==['127.0.0.1:18080:8000']
key=cfg['services']['api']['environment']['SUPABASE_SERVICE_KEY']
state=json.loads((root/'state.private.json').read_text())
source=Path(state['backup'])/'restored/objects'
# Compose can return before PostgREST/Storage have accepted connections. Probe a
# read-only authenticated endpoint before importing; never retry arbitrary writes.
ready = False
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
 try:
  request=urllib.request.Request('http://127.0.0.1:18080/storage/v1/bucket',headers={
    'Authorization':'Bearer '+key,'apikey':key})
  with urllib.request.urlopen(request,timeout=5) as response:
   ready=response.status == 200
  if ready:break
 except OSError:pass
 time.sleep(2)
if not ready:
 print(json.dumps({'stage':'storage_readiness','ready':False}));sys.exit(1)
started=time.monotonic();count=total=0
for file in sorted(source.rglob('*')):
 if not file.is_file():continue
 name=file.relative_to(source).as_posix();raw=file.read_bytes()
 request=urllib.request.Request('http://127.0.0.1:18080/storage/v1/object/'+urllib.parse.quote(name,safe='/'),data=raw,method='POST',headers={
   'Authorization':'Bearer '+key,'apikey':key,'x-upsert':'true',
   'Content-Type':mimetypes.guess_type(file.name)[0] or 'application/octet-stream'})
 try:
  with urllib.request.urlopen(request,timeout=120) as response:assert response.status in (200,201)
 except Exception as error:
  print(json.dumps({'stage':'storage_import','completed':count,'error_type':type(error).__name__,'http':getattr(error,'code',None)}));sys.exit(1)
 count+=1;total+=len(raw)
result={'objects_imported':count,'bytes':total,'seconds':round(time.monotonic()-started,3)}
(root/'storage-result.json').write_text(json.dumps(result));print(json.dumps(result))
