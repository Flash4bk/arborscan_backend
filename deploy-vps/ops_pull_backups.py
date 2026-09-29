"""Windows pull-only off-site transfer. No credentials or private data in logs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from ops_verify_offsite import verify

REMOTE = '/home/arborscan/ops-backups'
NAME = re.compile(r'\d{8}T\d{6}Z')
INVENTORY = '''import pathlib,json,re,hashlib
root=pathlib.Path('/home/arborscan/ops-backups')
result=[]
for p in sorted(root.iterdir()):
 if re.fullmatch(r'\\d{8}T\\d{6}Z',p.name) and (p/'COMPLETE').is_file() and (p/'SHA256SUMS').is_file():
  manifest=(p/'SHA256SUMS').read_text();blocks={}
  for line in manifest.splitlines():
   digest,name=line.split('  ',1);name=str(pathlib.PurePosixPath(name));file=p/name
   if file.stat().st_size>32*1024**2:
    parts=[];offset=0
    with file.open('rb') as stream:
     while raw:=stream.read(4*1024**2):
      parts.append({'offset':offset,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});offset+=len(raw)
    blocks[name]=parts
  result.append({'name':p.name,'manifest':manifest,'blocks':blocks,'bytes':sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and 'restored' not in f.relative_to(p).parts)})
print(json.dumps(result))
'''


def entries(manifest):
    result = {}
    for line in manifest.splitlines():
        digest, name = line.split('  ', 1)
        p = PurePosixPath(name)
        if (not re.fullmatch('[a-f0-9]{64}', digest) or p.is_absolute() or '..' in p.parts
                or not re.fullmatch(r'[A-Za-z0-9_./-]+', name) or str(p) in result):
            raise ValueError('invalid_manifest')
        result[str(p)] = digest
    if not {'application.tar', 'local-files.tar', 'arborscan.env.private', 'containers.private.json'} <= result.keys():
        raise ValueError('missing_required_archives')
    return result


def matches(path, digest):
    if not path.is_file():
        return False
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() == digest


def transfer(root, record, download, free=shutil.disk_usage, download_many=None, delta=None):
    name = record['name']
    if not NAME.fullmatch(name):
        raise ValueError('invalid_backup_name')
    files = entries(record['manifest'])
    native=('database.dump','roles.sql','source.json','archive-list.private.txt')
    if 'postgres/database.dump' in files and not all('postgres/'+n in files for n in (*native,'COMPLETE')):
        raise ValueError('incomplete_postgres_set')
    final = root / name
    if final.exists():
        # Older manually copied sets may lack our COMPLETE; verify before marking.
        if (final/'SHA256SUMS').read_text() != record['manifest']:
            raise ValueError('existing_set_manifest_changed')
        (final/'COMPLETE').unlink(missing_ok=True)
        verify(final)
        if 'postgres/database.dump' in files:
            (final/'postgres/SHA256SUMS').write_text(''.join(files['postgres/'+n]+'  '+n+'\n' for n in native))
        (final/'COMPLETE').touch()
        return 'verified_existing'
    partial = root / (name+'.partial')
    partial.mkdir(exist_ok=True)
    if free(root).free < int(record['bytes']) + 2*1024**3:
        raise ValueError('insufficient_disk')
    pending=[]
    for filename, digest in files.items():
        target = partial/filename
        if not target.resolve().is_relative_to(partial.resolve()):
            raise ValueError('unsafe_destination')
        target.parent.mkdir(parents=True, exist_ok=True)
        if matches(target, digest):
            continue
        if delta and delta(record,filename,target):
            if not matches(target,digest):raise ValueError('checksum_mismatch')
            # Only redundant staging bytes, after full-file verification. Never
            # remove a completed backup or the source used for matching blocks.
            target.with_name(target.name+'.downloading').unlink(missing_ok=True)
            continue
        temporary = target.with_name(target.name+'.downloading')
        if matches(temporary,digest):
            os.replace(temporary,target)
            continue
        pending.append((filename,temporary))
    if download_many and pending:
        download_many(name,pending)
    for filename,temporary in pending:
        digest=files[filename];target=partial/filename
        if not download_many:download(name, filename, temporary)
        if not matches(temporary, digest):
            quarantine=root/'transfer-errors';quarantine.mkdir(exist_ok=True)
            temporary.rename(quarantine/(name+'-'+str(time.time_ns())+'.bad-sha'))
            raise ValueError('checksum_mismatch')
        os.replace(temporary, target)
    (partial/'SHA256SUMS').write_text(record['manifest'])
    result = verify(partial)
    if 'postgres/database.dump' in files:
        # Parent manifest covers native files; native manifest may be omitted by
        # outer find. The required four native files must all be present.
        (partial/'postgres/SHA256SUMS').write_text(''.join(files['postgres/'+n]+'  '+n+'\n' for n in native))
        result['scope'] = 'one application/configuration/native PostgreSQL backup set'
    (partial/'OFFSITE_VERIFIED.json').write_text(json.dumps(result,indent=2))
    (partial/'COMPLETE').touch()
    partial.rename(final)
    return 'downloaded_verified'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default=r'D:\ArborScanBackups')
    parser.add_argument('--host',default='arborscan@31.57.170.88')
    parser.add_argument('--port',type=int,default=22)
    args=parser.parse_args()
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (root/'pull.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            with (root/'pull.log').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'skipped_already_running'})+'\n')
            return 0
        def log(event, **fields):
            with (root/'pull.log').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':event,**fields})+'\n')
        system=Path(os.environ['WINDIR'])/'System32/OpenSSH'
        options=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
        try:
            log('started')
            from ops_windows_job import contain_children
            contain_children()
            hidden={'creationflags':subprocess.CREATE_NO_WINDOW}
            inventory=subprocess.run([str(system/'ssh.exe'),*options,'-p',str(args.port),args.host,'python3 -'],input=INVENTORY.encode(),capture_output=True,timeout=180,**hidden)
            if inventory.returncode:raise RuntimeError('ssh_inventory_failed')
            records=json.loads(inventory.stdout)
            def delta(record,filename,target):
                import gzip
                from ops_delta_copy import reconstruct
                def fetch(chunks):
                    request={'backup':record['name'],'file':filename,'sha256':entries(record['manifest'])[filename],'chunks':chunks}
                    result=subprocess.run([str(system/'ssh.exe'),*options,'-p',str(args.port),args.host,
                        'python3 /home/arborscan/ops-tools/ops_backup_chunks.py'],input=json.dumps(request).encode(),capture_output=True,timeout=180,**hidden)
                    if result.returncode:raise RuntimeError('transfer_interrupted')
                    return gzip.decompress(result.stdout)
                return reconstruct(root,record,filename,target,fetch)
            def download(name, filename, target):
                result=subprocess.run([str(system/'scp.exe'),*options,'-P',str(args.port),f'{args.host}:{REMOTE}/{name}/{filename}',str(target)],capture_output=True,timeout=1800,**hidden)
                if result.returncode:raise RuntimeError('transfer_interrupted')
            def download_many(name, files):
                batch=''.join(f'{"reget" if p.exists() and p.stat().st_size else "get"} "{REMOTE}/{name}/{f}" "{p.as_posix()}"\n' for f,p in files)
                # One SSH connection per set, avoiding many short SCP sessions.
                batchfile=root/'transfer-batch.private.txt';batchfile.write_text(batch,encoding='utf-8')
                trace=root/'transfer.private.log'
                with trace.open('wb') as output:
                    process=subprocess.Popen([str(system/'sftp.exe'),*options,'-R','128','-B','131072','-P',str(args.port),'-b',str(batchfile),args.host],stdout=output,stderr=output,stdin=subprocess.DEVNULL,**hidden)
                    progress=time.monotonic();previous=-1;started=progress;reported=progress
                    while process.poll() is None:
                        size=sum(p.stat().st_size for _,p in files if p.exists())
                        if size!=previous:previous=size;progress=time.monotonic()
                        if time.monotonic()-reported>30:
                            log('transfer_progress',bytes=size);reported=time.monotonic()
                        if time.monotonic()-progress>90 or time.monotonic()-started>7200:
                            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,**hidden)
                            process.wait(timeout=15)
                            raise RuntimeError('transfer_stalled')
                        time.sleep(2)
                if process.returncode:
                    diagnostics=trace.read_text(errors='replace').lower()
                    log('sftp_failed',exit_code=process.returncode,categories=[x for x in
                        ('connection closed','connection reset','broken pipe','timed out','permission denied','no such file','bad message','failure') if x in diagnostics])
                    raise RuntimeError('transfer_interrupted')
            for record in records:
                for attempt in range(3):
                    try:
                        outcome=transfer(root,record,download,download_many=download_many,delta=delta)
                        break
                    except RuntimeError as error:
                        if str(error) not in ('transfer_interrupted','transfer_stalled') or attempt==2:raise
                        log('retry_transfer',backup=record['name'],attempt=attempt+2)
                        time.sleep(5)
                log(outcome,backup=record['name'])
            log('success',complete_sets=len(records))
            return 0
        except Exception as error:
            # Never print subprocess output, connection details, or private paths.
            safe=str(error) if str(error) in ('ssh_inventory_failed','transfer_interrupted',
                'transfer_stalled','checksum_mismatch','insufficient_disk','existing_set_manifest_changed') else 'validation_or_local_error'
            log('failed',error_type=type(error).__name__,reason=safe)
            return 1


if __name__=='__main__':
    sys.exit(main())
