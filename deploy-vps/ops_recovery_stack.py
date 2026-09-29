"""Build an isolated ArborScan recovery lab from one verified backup set.

No production env or libpq credentials are mounted. All service credentials are new.
Network is Docker-internal; only localhost HTTP ports. Never modifies production.
"""
import base64
import argparse
import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import time

BACKUP = Path('/home/arborscan/ops-backups/20260928T133059Z')
IMAGES = {'db':'supabase/postgres@sha256:178f0976b54a39237096bfa310c1a352dbc82fb1b08dda45cdb8acb5d40c1426',
          'rest':'postgrest/postgrest@sha256:c9dc201e555f5d8e37e7f39cdd4df0229774996e213bfd7de8d10ac609030f2c',
          'storage':'supabase/storage-api@sha256:b393ac5759a45a934557a150ecdb97157bbdd419e0193e4c5dc5770b9ddd7ee1',
          'gateway':'nginx@sha256:a8b39bd9cf0f83869a2162827a0caf6137ddf759d50a171451b335cecc87d236',
          'v3':'sha256:5a0b1d63e6c5eb13d14af84590dafebf900f59156613b987000ae8b6938f563b',
          'api':'sha256:afb6c57845667f9aff8a1e9236fbe7b81255f2c8ccc53ccc5b7b10cb77abb26e'}


def main():
    global BACKUP, IMAGES
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup', type=Path, default=BACKUP)
    parser.add_argument('--images-manifest', type=Path)
    args = parser.parse_args()
    BACKUP = args.backup.resolve()
    if args.images_manifest:
        candidate = json.loads(args.images_manifest.read_text())['images']
        if set(candidate) != set(IMAGES) or any(
            not __import__('re').fullmatch(r'sha256:[0-9a-f]{64}', value)
            for value in candidate.values()
        ):
            raise ValueError('Invalid pinned recovery image manifest')
        IMAGES = candidate
    os.umask(0o077)
    assert (BACKUP/'COMPLETE').is_file() and (BACKUP/'postgres/COMPLETE').is_file()
    # BusyBox (clean Docker-in-Docker bootstrap) lacks GNU --quiet. Capture
    # output instead so both implementations verify without exposing names.
    subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=BACKUP,check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path('/home/arborscan')/('as14-recovery-'+stamp);root.mkdir(mode=0o700)
    started=time.monotonic()
    def run(args,data=None):
        with (root/'diagnostics.private.log').open('ab') as log:
            p=subprocess.run(args,input=data,stdout=subprocess.PIPE,stderr=log,timeout=600)
        if p.returncode:raise RuntimeError('Recovery command failed; private diagnostics retained')
        return p.stdout
    pinned={k:run(['docker','image','inspect',v,'--format','{{.Id}}']).decode().strip() for k,v in IMAGES.items()}
    saved=json.loads((BACKUP/'containers.private.json').read_text())
    assert any(r['Name']=='/arborscan-api-v4' and r['Image']==pinned['api'] for r in saved)
    secret=secrets.token_hex(32)
    dbpass=secrets.token_hex(32)
    def jwt(role):
        enc=lambda b:base64.urlsafe_b64encode(b).rstrip(b'=')
        msg=enc(b'{"alg":"HS256","typ":"JWT"}')+b'.'+enc(json.dumps({'role':role,'iss':'as14-recovery','exp':int(time.time())+86400*30}).encode())
        return (msg+b'.'+enc(hmac.new(secret.encode(),msg,hashlib.sha256).digest())).decode()
    service=jwt('service_role');anon=jwt('anon')
    (root/'models').mkdir()
    with tarfile.open(BACKUP/'local-files.tar') as tar:
        for m in tar:
            if m.isfile() and m.name.startswith('opt/arborscan/models/'):
                name=Path(m.name).name
                if name.endswith('.pt'):
                    (root/'models'/name).write_bytes(tar.extractfile(m).read())
    for p in (root/'models').iterdir():p.chmod(0o644)
    (root/'models').chmod(0o755)
    (root/'gateway.conf').write_text('''events {}\nhttp {
      access_log off; server { listen 8000; client_max_body_size 256m;
      location /rest/v1/ { proxy_pass http://rest:3000/; }
      location /storage/v1/ { proxy_pass http://storage:5000/; }
      }}''')
    env={'SUPABASE_URL':'http://gateway:8000','SUPABASE_SERVICE_KEY':service,
         'PROJECT_ROOT':'/app','MODEL_DIR':'/app/models','ACTIVE_MODEL_VERSION':'4',
         'AUTO_SELECT_LATEST_LOCAL_MODEL':'false','MODEL_QUALITY_ENABLED':'1',
         'MODEL_QUALITY_DIR':'/app/model-quality','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','YOLO_AUTOINSTALL':'false','REMBG_PRELOAD':'false',
         'SUPABASE_ENABLE_QUEUE':'false'}
    common={'restart':'no','logging':{'driver':'json-file','options':{'max-size':'5m','max-file':'2'}}}
    app=lambda key,port:{**common,'image':pinned[key],'environment':env,
       'ports':[f'127.0.0.1:{18000 if key=="v3" else 18001}:{port}'],
       'volumes':[str(root/'models')+':/app/models:ro','runtime:/app/model-quality'],
       'mem_limit':'2g','cpus':1}
    services={
      'db':{**common,'image':pinned['db'],'user':'postgres','entrypoint':['/bin/bash','-c'],
        'command':['test -f /var/lib/postgresql/data/as14/PG_VERSION || initdb -D /var/lib/postgresql/data/as14 -U supabase_admin --auth=trust; exec postgres -D /var/lib/postgresql/data/as14 -c listen_addresses=* -c shared_preload_libraries='],
        'volumes':['db:/var/lib/postgresql/data'],'mem_limit':'2g','cpus':1},
      'rest':{**common,'image':pinned['rest'],'environment':{'PGRST_DB_URI':f'postgres://authenticator:{dbpass}@db:5432/postgres','PGRST_DB_SCHEMAS':'public','PGRST_DB_ANON_ROLE':'anon','PGRST_JWT_SECRET':secret},'mem_limit':'256m'},
      'storage':{**common,'image':pinned['storage'],'environment':{
        'ANON_KEY':anon,'SERVICE_KEY':service,'AUTH_JWT_SECRET':secret,'POSTGREST_URL':'http://rest:3000',
        'DATABASE_URL':f'postgres://supabase_storage_admin:{dbpass}@db:5432/postgres','STORAGE_BACKEND':'file',
        'GLOBAL_S3_BUCKET':'stub','FILE_STORAGE_BACKEND_PATH':'/var/lib/storage','TENANT_ID':'stub',
        'REGION':'stub','FILE_SIZE_LIMIT':'268435456','ENABLE_IMAGE_TRANSFORMATION':'false'},
        'volumes':['storage:/var/lib/storage'],'mem_limit':'512m'},
      'gateway':{**common,'image':pinned['gateway'],'networks':['default','edge'],'volumes':[str(root/'gateway.conf')+':/etc/nginx/nginx.conf:ro'],'ports':['127.0.0.1:18080:8000'],'mem_limit':'128m'},
      'v3':app('v3',8000),'api':app('api',8001),
      'worker':{**common,'image':pinned['api'],'environment':env,'command':['python','-m','arborscan_v4.quality_worker'],
        'healthcheck':{'test':['CMD','python','-c',"import pathlib,time; p=pathlib.Path('/app/model-quality/heartbeat'); assert p.exists() and time.time()-p.stat().st_mtime<45"],'interval':'15s','timeout':'5s','start_period':'30s','retries':3},
        'volumes':[str(root/'models')+':/app/models:ro','runtime:/app/model-quality'],'mem_limit':'1g','cpus':1}}
    config={'name':'as14-recovery-'+stamp.lower(),'services':services,
        'networks':{'default':{'internal':True},'edge':{}},'volumes':{'db':{},'storage':{},'runtime':{}}}
    compose=root/'compose.private.json';compose.write_text(json.dumps(config))
    cmd=['docker','compose','-f',str(compose)]
    run(['docker','volume','create',config['name']+'_db'])
    run(['docker','run','--rm','--network','none','--user','root',
         '-v',config['name']+'_db:/restore-db','--entrypoint','sh',pinned['db'],
         '-c','chown postgres:postgres /restore-db'])
    run(cmd+['up','-d','db'])
    for _ in range(30):
        try:run(cmd+['exec','-T','db','pg_isready','-U','supabase_admin']);break
        except RuntimeError:time.sleep(1)
    sql=cmd+['exec','-T','db','psql','-U','supabase_admin','-d','postgres','-v','ON_ERROR_STOP=1','-At']
    roles=(BACKUP/'postgres/roles.sql').read_text()
    definitions='\n'.join(x for x in roles.splitlines() if not x.startswith('GRANT ') and x!='CREATE ROLE supabase_admin;')
    grants='\n'.join(x for x in roles.splitlines() if x.startswith('GRANT '))
    run(sql,(definitions+'\n'+grants).encode())
    run(cmd+['exec','-T','db','pg_restore','-U','supabase_admin','-d','postgres','--exit-on-error','--single-transaction'],(BACKUP/'postgres/database.dump').read_bytes())
    # The copied database contains old credentials/sessions; isolate and invalidate
    # sessions in the COPY only. No real queued job may execute in this lab.
    run(sql,b"DELETE FROM public.auth_sessions; UPDATE public.ml_jobs SET state='failed' WHERE state IN ('queued','running','cancel_requested');")
    run(sql,f"ALTER ROLE authenticator PASSWORD '{dbpass}'; ALTER ROLE supabase_storage_admin PASSWORD '{dbpass}';".encode())
    run(cmd+['exec','-T','db','sh','-c','echo "host all all all scram-sha-256" >> /var/lib/postgresql/data/as14/pg_hba.conf'])
    run(sql,b'SELECT pg_reload_conf();')
    run(cmd+['up','-d','rest','storage','gateway'])
    (root/'state.private.json').write_text(json.dumps({'root':str(root),'backup':str(BACKUP),'images':pinned,'started_unix':time.time(),'setup_seconds':time.monotonic()-started}))
    print(json.dumps({'root':str(root),'database_restored':True,'images':pinned}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('Recovery setup incomplete: '+type(error).__name__+'; inspect private diagnostics')
        sys.exit(1)
