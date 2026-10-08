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
INVENTORY = '''import pathlib,json,re,hashlib,subprocess
root=pathlib.Path('/home/arborscan/ops-backups')
result=[]
protected=set()
live=json.loads(subprocess.run(['docker','inspect','arborscan-api','arborscan-api-v4','arborscan-quality-worker'],capture_output=True,text=True,check=True).stdout)
for row in live:
 paths=row['Config'].get('Labels',{}).get('com.docker.compose.project.config_files','').split(',')+[m['Source'] for m in row.get('Mounts',[])]
 for path in paths:
  match=re.search(r'/ops-backups/(\\d{8}T\\d{6}Z)(?:/|$)',path)
  if match:protected.add(match.group(1))
runtime_checked={}
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
  record={'name':p.name,'manifest':manifest,'blocks':blocks,'bytes':sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and 'restored' not in f.relative_to(p).parts),'protected_sets':sorted(protected)}
  runtime_file=p/'RUNTIME_DEPENDENCIES.json'
  if runtime_file.is_file():
   expected=dict((name,digest) for digest,name in (line.split('  ',1) for line in manifest.splitlines())).get('RUNTIME_DEPENDENCIES.json')
   if hashlib.sha256(runtime_file.read_bytes()).hexdigest()!=expected:raise ValueError('runtime_contract_not_verified')
   runtime=json.loads(runtime_file.read_bytes());record['runtime']=runtime;assets={}
   for items in runtime['images'].values():
    for asset in items:
     path=pathlib.Path(asset['path']);digest=asset['sha256']
     if not path.is_absolute() or path.is_symlink() or any(x.is_symlink() for x in path.parents) or not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('unsafe_runtime_archive')
     if path not in runtime_checked:
      with path.open('rb') as stream:runtime_checked[path]=hashlib.file_digest(stream,'sha256').hexdigest()
     if runtime_checked[path]!=digest:raise ValueError('runtime_archive_checksum_mismatch')
     assets[str(path)]={'path':str(path),'sha256':digest,'bytes':path.stat().st_size}
   record['runtime_assets']=list(assets.values())
  result.append(record)
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


def read_inventory(command, *, hidden=None, log=None, run=None, sleep=None):
    """At most three read-only attempts; never log private subprocess output.

    Each SSH invocation is bounded to 180 seconds. Authentication and host-key
    arguments remain identical; retries cannot authorize an untrusted peer.
    Invalid successful responses are validation failures, not transport retries.
    """
    run = subprocess.run if run is None else run
    sleep = time.sleep if sleep is None else sleep
    for attempt in range(1, 4):
        try:
            result = run(command, input=INVENTORY.encode(), capture_output=True,
                         timeout=180, **(hidden or {}))
        except subprocess.TimeoutExpired:
            fields = {'categories': ['timeout']}
        except OSError:
            if log:
                log('inventory_attempt_failed', attempt=attempt,
                    categories=['local_client_error'])
            raise RuntimeError('ssh_inventory_failed') from None
        else:
            if result.returncode == 0:
                try:
                    records = json.loads(result.stdout)
                    if not isinstance(records, list) or not records:
                        raise ValueError('invalid inventory')
                except (ValueError, TypeError, UnicodeError):
                    raise RuntimeError('ssh_inventory_invalid_response') from None
                return records
            diagnostics = result.stderr or b''
            if isinstance(diagnostics, bytes):
                diagnostics = diagnostics.decode('utf-8', errors='replace')
            diagnostics = diagnostics.lower()
            categories = [category for text, category in (
                ('connection closed', 'connection_closed'),
                ('connection reset', 'connection_reset'),
                ('connection refused', 'connection_refused'),
                ('timed out', 'timeout'),
                ('permission denied', 'authentication_denied'),
                ('host key verification failed', 'host_key_rejected'),
                ('could not resolve hostname', 'name_resolution_failed'),
                ('network is unreachable', 'network_unreachable')) if text in diagnostics]
            fields = {'exit_code': result.returncode,
                      'categories': categories or ['ssh_nonzero_exit']}
        if log:
            log('inventory_attempt_failed', attempt=attempt, **fields)
        if attempt == 3:
            raise RuntimeError('ssh_inventory_failed') from None
        sleep(5 * attempt)


def transfer_runtime(root, folder, record, download_many, free=shutil.disk_usage):
    """One SHA-addressed shared store; original covered dependencies stay unchanged."""
    if 'runtime' not in record:
        return None
    from ops_windows_retention import safe_path, verify_runtime
    runtime_file = folder/'RUNTIME_DEPENDENCIES.json'
    if json.loads(runtime_file.read_bytes()) != record['runtime']:
        raise ValueError('runtime_contract_changed')
    shared = safe_path(root, root/'runtime-assets')
    shared.mkdir(exist_ok=True)
    assets = record.get('runtime_assets', [])
    expected = {(a['path'], a['sha256']) for items in record['runtime']['images'].values() for a in items}
    if expected != {(a['path'], a['sha256']) for a in assets}:
        raise ValueError('runtime_asset_inventory_mismatch')
    pending = []
    relocated = []
    required_bytes = 0
    checked = set()
    for asset in assets:
        source = asset['path']; digest = asset['sha256']
        if (not re.fullmatch(r'/home/arborscan/[A-Za-z0-9_./-]+', source) or
                '..' in PurePosixPath(source).parts or not re.fullmatch('[a-f0-9]{64}', digest)):
            raise ValueError('unsafe_runtime_archive')
        target = safe_path(root, shared/(digest+'.archive'))
        relocated.append({'original_path':source,'restored_path':str(target),'sha256':digest})
        if digest in checked:
            continue
        checked.add(digest)
        if matches(target, digest):
            continue
        # Reuse a previously verified relocation asset with a hard link. Shared
        # bytes remain available if a redundant recovery directory is removed.
        candidates = list(root.glob('*-current-runtime-recovery.private/runtime-assets/'+digest+'.archive'))
        for candidate in candidates:
            safe_path(root, candidate)
            if matches(candidate, digest):
                link = safe_path(root, target.with_name(target.name+'.reusing'))
                if link.exists():
                    link.unlink()
                os.link(candidate, link)
                os.replace(link, target)
                break
        if matches(target, digest):
            continue
        temporary = safe_path(root, target.with_name(target.name+'.downloading'))
        if matches(temporary, digest):
            os.replace(temporary, target)
            continue
        size = int(asset['bytes'])
        if size < 0 or (temporary.exists() and temporary.stat().st_size > size):
            raise ValueError('invalid_runtime_size')
        required_bytes += size - (temporary.stat().st_size if temporary.exists() else 0)
        pending.append((source, temporary))
    if pending:
        if free(root).free < required_bytes + 2*1024**3:
            raise ValueError('insufficient_disk')
        if download_many is None:
            raise ValueError('runtime_transfer_required')
        download_many(None, pending)
        for source, temporary in pending:
            asset = next(a for a in assets if a['path'] == source)
            if not matches(temporary, asset['sha256']):
                # A failed resume cannot be retried as a complete asset.
                temporary.unlink(missing_ok=True)
                raise ValueError('checksum_mismatch')
            os.replace(temporary, shared/(asset['sha256']+'.archive'))
    mapping = folder/'RELOCATED_RUNTIME.json'
    writing = safe_path(root, mapping.with_name(mapping.name+'.writing'))
    writing.write_text(json.dumps({'format':1,'assets':relocated}),encoding='utf-8')
    os.replace(writing, mapping)
    return verify_runtime(root, folder)


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
        try:
            verify(final)
        except (ValueError, OSError):
            # Revoke a genuinely damaged set, not a valid set while checking.
            (final/'COMPLETE').unlink(missing_ok=True)
            raise
        transfer_runtime(root, final, record, download_many, free)
        if 'postgres/database.dump' in files and 'postgres/SHA256SUMS' not in files:
            (final/'postgres/SHA256SUMS').write_text(''.join(files['postgres/'+n]+'  '+n+'\n' for n in native), newline='\n')
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
    # Preserve the trusted SSH manifest byte for byte on Windows as well.
    # Translating LF to CRLF would change its hash used by external receipts.
    (partial/'SHA256SUMS').write_text(record['manifest'], encoding='utf-8', newline='\n')
    result = verify(partial)
    transfer_runtime(root, partial, record, download_many, free)
    if 'postgres/database.dump' in files and 'postgres/SHA256SUMS' not in files:
        # Parent manifest covers native files; native manifest may be omitted by
        # outer find. The required four native files must all be present.
        (partial/'postgres/SHA256SUMS').write_text(''.join(files['postgres/'+n]+'  '+n+'\n' for n in native), newline='\n')
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
    from ops_windows_retention import canonical_root
    root=canonical_root(args.root)
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
            records=read_inventory([str(system/'ssh.exe'),*options,'-p',str(args.port),
                                    args.host,'python3 -'],hidden=hidden,log=log)
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
                lines=[]
                for filename, destination in files:
                    remote_path = filename if name is None else REMOTE+'/'+name+'/'+filename
                    operation = 'reget' if destination.exists() and destination.stat().st_size else 'get'
                    lines.append(f'{operation} "{remote_path}" "{destination.as_posix()}"\n')
                batch=''.join(lines)
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
            # Rotation is available only after an all-three-service native
            # replacement and SHA-addressed runtime readback. Old unadopted
            # manual/historical sets and live Compose dependencies remain pinned.
            latest = records[-1]
            if set(latest.get('runtime',{}).get('services',[])) == {
                    '/arborscan-api','/arborscan-api-v4','/arborscan-quality-worker'}:
                from ops_windows_retention import plan, apply
                proposal = plan(root, latest['name'], latest.get('protected_sets', []))
                (root/'retention-dry-run.private.json').write_text(json.dumps(proposal,indent=2))
                rotation = apply(root, proposal, latest.get('protected_sets', []), lock_held=True)
                log('retention_complete', **rotation)
            log('success',complete_sets=len(records))
            return 0
        except Exception as error:
            # Never print subprocess output, connection details, or private paths.
            safe=str(error) if str(error) in ('ssh_inventory_failed','ssh_inventory_invalid_response','transfer_interrupted',
                'transfer_stalled','checksum_mismatch','insufficient_disk','existing_set_manifest_changed') else 'validation_or_local_error'
            log('failed',error_type=type(error).__name__,reason=safe)
            return 1


if __name__=='__main__':
    sys.exit(main())
