"""Start recovered application images; no production environment or training."""
import json,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert root.parent==Path('/home/arborscan') and root.name.startswith('as14-recovery-')
path=root/'compose.private.json';c=json.loads(path.read_text())
assert c['networks']['default']['internal'] is True
cmd=['docker','compose','-f',str(path)]
with (root/'diagnostics.private.log').open('ab') as log:
 def run(args):return subprocess.run(args,stdout=log,stderr=log,check=True,timeout=300)
 run(['docker','volume','create',c['name']+'_runtime'])
 run(['docker','run','--rm','--network','none','--user','root','-v',c['name']+'_runtime:/runtime','--entrypoint','sh',c['services']['api']['image'],'-c','chown 1000:1000 /runtime'])
 run(cmd+['up','-d','v3','api'])
 conf=(root/'gateway.conf').read_text()
 if 'location /api/v3/' not in conf:
  conf=conf.replace('      }}','''      location /api/v3/ { proxy_pass http://v3:8000/; }
      location /api/v4/ { proxy_pass http://api:8001/; }
      }}''')
  (root/'gateway.conf').write_text(conf)
 run(cmd+['exec','-T','gateway','nginx','-s','reload'])
 for version in ('v3','v4'):
  for _ in range(90):
   try:
    with urllib.request.urlopen('http://127.0.0.1:18080/api/'+version+'/health',timeout=5) as r:
     assert json.load(r)['status']=='ok'
    break
   except Exception:time.sleep(2)
  else:raise RuntimeError('Recovered API readiness failed')
print('Recovered v3/v4 HTTP health OK on localhost:18080')
