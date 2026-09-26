"""Explicit pinned rollout using privately captured compose paths; no env printing."""
import json
import os
import pathlib
import subprocess
import sys

backup=pathlib.Path('/home/arborscan/ops-backups/20260926T202023Z')
image='arborscan-api-v4:reliability-691fb77'
mode=sys.argv[1]
assert mode in ('prepare','deploy','rollback')
records=json.loads((backup/'containers.private.json').read_text())
old={c['Name'].lstrip('/'):c for c in records}
overlay=backup/'reliability.yml'
rollback=backup/'rollback.yml'
overlay.write_text('services:\n  api-v4:\n    image: '+image+'\n  quality-worker:\n    image: '+image+'\n')
rollback.write_text('services:\n  api-v4:\n    image: '+old['arborscan-api-v4']['Image']+'\n  quality-worker:\n    image: '+old['arborscan-quality-worker']['Image']+'\n')
if mode=='prepare':
    print('Prepared exact old-image rollback and candidate overlay');sys.exit(0)
subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=backup,check=True)
if mode=='deploy':
    for name in old:
        current=subprocess.check_output(['docker','inspect','-f','{{.Image}}',name],text=True).strip()
        assert current==old[name]['Image'],'Runtime changed; re-audit required'
    subprocess.run(['docker','exec','arborscan-api-v4','python','-c',
        'from arborscan_v4.model_quality_api import QualityStore; assert not QualityStore().rows("ml_jobs",state="in.(queued,running,cancel_requested)",select="id"), "Active jobs; postpone rollout"'],check=True)
cmd=['docker','compose','-p','arborscan-v4']
for path in old['arborscan-api-v4']['Config']['Labels']['com.docker.compose.project.config_files'].split(','):
    cmd+=['-f',path]
cmd+=['-f',str(overlay if mode=='deploy' else rollback),'up','-d','--no-build','--no-deps','api-v4','quality-worker']
subprocess.run(cmd,env={**os.environ,'MODEL_QUALITY_IMAGE':image},check=True)
print(mode+' completed; verify health, readiness and authenticated fixtures')
