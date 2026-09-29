"""Sanitized evidence for a recovery lab, never dump Docker environment values."""
import datetime,json,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(sys.argv[1]).resolve();assert root.parent==Path('/home/arborscan') and root.name.startswith('as14-recovery-')
path=root/'compose.private.json';c=json.loads(path.read_text());assert c['networks']['default']['internal']
c['services']['worker']['healthcheck']={'test':['CMD','python','-c',"import pathlib,time; p=pathlib.Path('/app/model-quality/heartbeat'); assert p.exists() and time.time()-p.stat().st_mtime<45"],'interval':'15s','timeout':'5s','start_period':'30s','retries':3}
path.write_text(json.dumps(c));cmd=['docker','compose','-f',str(path)]
with (root/'diagnostics.private.log').open('ab') as log:
 subprocess.run(cmd+['up','-d','worker'],stdout=log,stderr=log,check=True)
time.sleep(20)
ids=subprocess.check_output(cmd+['ps','-q'],text=True).split()
containers=json.loads(subprocess.check_output(['docker','inspect',*ids]))
summary=[]
for row in containers:
 service=row['Config']['Labels']['com.docker.compose.service']
 assert row['State']['Running']
 if service in ('api','v3','db','worker'):assert row['State'].get('Health',{}).get('Status')=='healthy'
 for mount in row['Mounts']:
  assert '.pgpass' not in mount['Source'] and '/etc/arborscan' not in mount['Source'] and not mount['Source'].startswith('/opt/arborscan')
 bindings=row['HostConfig'].get('PortBindings') or {}
 assert all(b['HostIp']=='127.0.0.1' for values in bindings.values() for b in values)
 summary.append({'service':service,'image':row['Image'],'running':True,'health':row['State'].get('Health',{}).get('Status'),'ports_localhost_only':True})
for v in ('v3','v4'):
 with urllib.request.urlopen('http://127.0.0.1:18080/api/'+v+'/health',timeout=15) as r:assert json.load(r)['status']=='ok'
test=b"import socket,json\ns=socket.socket();s.settimeout(3)\ntry:s.connect(('31.57.170.88',443));print('UNEXPECTED_EGRESS')\nexcept OSError:print('EGRESS_BLOCKED')\n"
check=subprocess.check_output(cmd+['exec','-T','api','python','-'],input=test).decode().strip()
assert check=='EGRESS_BLOCKED'
started=datetime.datetime.strptime(root.name.removeprefix('as14-recovery-'),'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc)
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'source_backup':json.loads((root/'state.private.json').read_text())['backup'],
 'services':summary,'storage':json.loads((root/'storage-result.json').read_text()),'api_egress':'blocked',
 'elapsed_including_diagnostics_seconds':round(time.time()-started.timestamp()),'guaranteed_rto':False,'production_reconfigured':False}
(root/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
