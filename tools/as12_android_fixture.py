"""Seed synthetic AS-12 local reports; default target is an isolated emulator.
Run from repo root after test/report_export_test.dart generates output/pdf/as12.
Phone mode reads cached owner in memory, preserves session and never writes to server.
"""
import json
import os
import pathlib
import subprocess
import sys
import re
import xml.etree.ElementTree as ET

ADB = str(pathlib.Path(os.environ['LOCALAPPDATA']) / 'Android/Sdk/platform-tools/adb.exe')
SERIAL = sys.argv[1] if len(sys.argv) > 1 else 'emulator-5580'
existing_account = '--existing-account' in sys.argv
if not SERIAL.startswith('emulator-') and not existing_account:
    raise SystemExit('Phone requires --existing-account; never replace its session')
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
if existing_account:
    # Read only the cached owner into memory; never log or write credentials.
    prefs_tree=ET.fromstring(adb('shell','run-as',PKG,'cat','shared_prefs/FlutterSharedPreferences.xml'))
    OWNER=next((n.text for n in prefs_tree if n.get('name')=='flutter.arborscan_user_id'),None)
    if not OWNER or not re.fullmatch('[0-9a-fA-F-]{36}',OWNER):
        raise SystemExit('No cached owner; account left unchanged')
prefs = f'''<?xml version="1.0" encoding="utf-8"?><map>
<string name="flutter.arborscan_auth_token">as12-demo-invalid-no-server-access</string>
<string name="flutter.arborscan_user_id">{OWNER}</string>
</map>'''
if not existing_account:
    write('shared_prefs/FlutterSharedPreferences.xml',prefs.encode())
source=pathlib.Path('output/pdf/as12')
for name in ['full-v1','full-v2','old','long']:
    data=json.loads((source/f'{name}.json').read_text(encoding='utf-8'))
    data={**data['record'],'snapshot':data['snapshot'],'saved':False}
    # Display an explicit fixture name in the existing local history date line.
    data['created_at']=f'DEMO AS-12 {name} / 2026-09-25'
    prefix=f'files/report-history-v1/contour-drafts-v1/{OWNER}/as12-{name}'
    if existing_account:
        check=subprocess.run([ADB,'-s',SERIAL,'shell','run-as',PKG,'test','-e',prefix+'.json'],capture_output=True)
        if check.returncode==0:
            raise SystemExit('Fixture name already exists; not overwriting')
    write(prefix+'.json',json.dumps(data,ensure_ascii=False).encode())
    write(prefix+'.photo',(source/'demo.photo').read_bytes())
adb('shell','am','start','-n',PKG+'/.MainActivity')
print('Seeded four DEMO local records; no server writes; existing phone session preserved' if existing_account else 'Seeded four DEMO records in isolated emulator; no server writes')
