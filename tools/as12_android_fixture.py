"""Seed only isolated emulator with synthetic AS-12 local reports, never production.
Run from repo root after test/report_export_test.dart generates output/pdf/as12.
No real credentials are read; report values are explicitly demonstration fixtures.
"""
import json
import os
import pathlib
import subprocess
import sys

ADB = str(pathlib.Path(os.environ['LOCALAPPDATA']) / 'Android/Sdk/platform-tools/adb.exe')
SERIAL = sys.argv[1] if len(sys.argv) > 1 else 'emulator-5580'
if not SERIAL.startswith('emulator-'):
    raise SystemExit('Fixture account allowed only on isolated emulator')
PKG = 'com.example.arborscan_app'
OWNER = '00000000-0000-4000-8000-000000001212'
def adb(*args, data=None):
    return subprocess.run([ADB, '-s', SERIAL, *args], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
def write(path, contents):
    temp=pathlib.Path(os.environ['TEMP'])/'as12-seed.bin'
    temp.write_bytes(contents)
    adb('push',str(temp),'/data/local/tmp/as12-seed.bin')
    adb('shell','run-as',PKG,'mkdir','-p',str(pathlib.PurePosixPath(path).parent))
    adb('shell','run-as',PKG,'cp','/data/local/tmp/as12-seed.bin',path)
    adb('shell','rm','/data/local/tmp/as12-seed.bin')
    temp.unlink()
adb('shell','am','force-stop',PKG)
prefs = f'''<?xml version="1.0" encoding="utf-8"?><map>
<string name="flutter.arborscan_auth_token">as12-demo-invalid-no-server-access</string>
<string name="flutter.arborscan_user_id">{OWNER}</string>
</map>'''
write('shared_prefs/FlutterSharedPreferences.xml',prefs.encode())
source=pathlib.Path('output/pdf/as12')
for name in ['full-v1','full-v2','old','long']:
    data=json.loads((source/f'{name}.json').read_text(encoding='utf-8'))
    data={**data['record'],'snapshot':data['snapshot'],'saved':False}
    # Display an explicit fixture name in the existing local history date line.
    data['created_at']=f'DEMO AS-12 {name} / 2026-09-25'
    prefix=f'files/report-history-v1/contour-drafts-v1/{OWNER}/as12-{name}'
    write(prefix+'.json',json.dumps(data,ensure_ascii=False).encode())
    write(prefix+'.photo',(source/'demo.photo').read_bytes())
adb('shell','am','start','-n',PKG+'/.MainActivity')
print('Seeded four DEMO records in isolated emulator; no server writes')
