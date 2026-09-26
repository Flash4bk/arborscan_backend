"""Create two marked server reports for the signed-in phone, without changing its session.
Credentials stay in memory. Uses ordinary authenticated report API, never admin writes.
"""
import copy
import json
import os
import pathlib
import subprocess
import sys
import uuid
import urllib.request
import xml.etree.ElementTree as ET

adb=str(pathlib.Path(os.environ['LOCALAPPDATA'])/'Android/Sdk/platform-tools/adb.exe')
serial=sys.argv[1] if len(sys.argv)>1 else 'R5CY40HNVCP'
prefs=ET.fromstring(subprocess.check_output([adb,'-s',serial,'shell','run-as',
    'com.example.arborscan_app','cat','shared_prefs/FlutterSharedPreferences.xml']))
token=next(n.text for n in prefs if n.get('name')=='flutter.arborscan_auth_token')
root=pathlib.Path('output/as14');root.mkdir(exist_ok=True)
target=root/('server-fixture-emulator.json' if serial.startswith('emulator-') else 'server-fixture.json')
if target.exists(): raise SystemExit('Existing fixture manifest: refusing duplicates')
photo=pathlib.Path('output/pdf/as12/demo.photo').read_bytes()
snapshot=json.loads(pathlib.Path('output/pdf/as12/full-v1.json').read_text(encoding='utf-8'))['snapshot']
snapshot['reference']['same_plane']=True
snapshot['reference']['version']=1
snapshot['reference']['method']='known_object_segment_v1'
for key in ['crown_height','trunk','trunk_axis']: snapshot['reference'].pop(key,None)
snapshot['report']={'species':'DEMO AS-14 server export'}
snapshot['captured_at']='2026-09-26T20:15:00Z'
snapshot['change_source']='reference'
analysis=str(uuid.uuid4());parent=None;created=[]
try:
    for length in [1,2]:
        version=str(uuid.uuid4());s=copy.deepcopy(snapshot);s['reference']['length_m']=length
        boundary=uuid.uuid4().hex
        fields={'analysis_id':analysis,'version_id':version,'snapshot':json.dumps(s)}
        if parent:fields['parent_id']=parent
        body=b''
        for name,value in fields.items():
            body+=f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        body+=f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="demo.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+photo+f'\r\n--{boundary}--\r\n'.encode()
        req=urllib.request.Request('https://31.57.170.88/api/v4/v4/reports',data=body,
            headers={'Authorization':'Bearer '+token,'Content-Type':'multipart/form-data; boundary='+boundary})
        with urllib.request.urlopen(req,timeout=120) as response:result=json.load(response)
        assert result['saved'] and result['record']['version_id']==version
        created.append(result['record']);parent=version
        target.write_text(json.dumps({'analysis_id':analysis,'records':created},indent=2),encoding='utf-8')
    print('Created two DEMO AS-14 versions through public authenticated API; no real report modified')
except Exception:
    print('Fixture creation failed; inspect HTTP status privately, never log credentials')
    raise SystemExit(1)
