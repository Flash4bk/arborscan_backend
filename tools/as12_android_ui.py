"""ADB UI evidence helper. Explicit serial; no recipient selection or data clearing."""
import os, pathlib, re, subprocess, sys, time, xml.etree.ElementTree as ET
adb=str(pathlib.Path(os.environ['LOCALAPPDATA'])/'Android/Sdk/platform-tools/adb.exe')
serial, action = sys.argv[1:3]
def run(*args):
    return subprocess.check_output([adb,'-s',serial,*args])
out=pathlib.Path('output/pdf/as12/device'); out.mkdir(parents=True,exist_ok=True)
if action=='screenshot':
    (out/(sys.argv[3]+'.png')).write_bytes(run('exec-out','screencap','-p'))
else:
    run('shell','uiautomator','dump','/sdcard/as12-ui.xml')
    xml=run('shell','cat','/sdcard/as12-ui.xml')
    nodes=list(ET.fromstring(xml).iter('node'))
    if action=='dump':
        for n in nodes:
            label=n.get('text') or n.get('content-desc')
            if label: print(label, n.get('bounds'))
    elif action in ['tap','wait']:
        matches=[n for n in nodes if sys.argv[3] in (n.get('text','')+' '+n.get('content-desc',''))]
        for attempt in range(6):
            if matches: break
            time.sleep(0.5)
            run('shell','uiautomator','dump','/sdcard/as12-ui.xml')
            nodes=list(ET.fromstring(run('shell','cat','/sdcard/as12-ui.xml')).iter('node'))
            matches=[n for n in nodes if sys.argv[3] in (n.get('text','')+' '+n.get('content-desc',''))]
        if not matches: raise SystemExit('UI label not found: '+sys.argv[3])
        if action=='wait':
            print('Confirmed UI: '+sys.argv[3]); sys.exit(0)
        n=matches[int(sys.argv[4]) if len(sys.argv)>4 else 0]
        x1,y1,x2,y2=map(int,re.findall(r'\d+',n.get('bounds')))
        run('shell','input','tap',str((x1+x2)//2),str((y1+y2)//2))
