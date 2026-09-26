"""Host diagnostic: public operational facts only, never env or user records."""
import json
import pathlib
import shutil
import subprocess
import time

def status():
    result={'disk_free_bytes':shutil.disk_usage('/home/arborscan').free,'containers':[]}
    for name in ('arborscan-api','arborscan-api-v4','arborscan-quality-worker'):
        p=subprocess.run(['docker','inspect',name],capture_output=True)
        if p.returncode:
            result['containers'].append({'name':name,'missing':True});continue
        c=json.loads(p.stdout)[0]
        result['containers'].append({'name':name,'image':c['Config']['Image'],'image_id':c['Image'],
            'running':c['State']['Running'],'health':c['State'].get('Health',{}).get('Status'),
            'started':c['State']['StartedAt'],'restarts':c['RestartCount'],
            'logging':c['HostConfig']['LogConfig']})
    root=pathlib.Path('/home/arborscan/ops-backups')
    completed=sorted(root.glob('*/COMPLETE'))
    result['latest_completed_backup']=str(completed[-1].parent) if completed else None
    result['backup_age_seconds']=round(time.time()-completed[-1].stat().st_mtime) if completed else None
    result['incomplete_backups']=[str(p) for p in root.glob('20*') if p.is_dir() and not (p/'COMPLETE').exists()]
    p=subprocess.run(['docker','exec','arborscan-api-v4','python','-c',
        'import json; from arborscan_v4.reliability import dependency_status; print(json.dumps(dependency_status()))'],
        capture_output=True,timeout=150)
    result['dependency_readiness']=json.loads(p.stdout) if p.returncode==0 else 'unavailable_on_this_runtime'
    print(json.dumps(result,indent=2))

if __name__=='__main__':status()
