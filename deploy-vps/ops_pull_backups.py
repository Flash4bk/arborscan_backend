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
known=globals().get('KNOWN_LOCAL_MANIFESTS',{})
protected=set()
ids=subprocess.run(['docker','ps','-aq'],capture_output=True,text=True,check=True).stdout.split()
if not ids:raise ValueError('live_container_inventory_empty')
live=json.loads(subprocess.run(['docker','inspect',*ids],capture_output=True,text=True,check=True).stdout)
for row in live:
 paths=(row['Config'].get('Labels') or {}).get('com.docker.compose.project.config_files','').split(',')+[m['Source'] for m in row.get('Mounts',[])]
 for path in paths:
  match=re.search(r'/ops-backups/(\\d{8}T\\d{6}Z)(?:/|$)',path)
  if match:protected.add(match.group(1))
runtime_checked={}
transition=root/'SELFHOST_TRANSITION_PROTECTION.private.json'
if transition.is_symlink():raise ValueError('unsafe_transition_protection')
protection=json.loads(transition.read_bytes()) if transition.is_file() else None
for p in sorted(root.iterdir()):
 if re.fullmatch(r'\\d{8}T\\d{6}Z',p.name) and (p/'COMPLETE').is_file() and (p/'SHA256SUMS').is_file():
  raw_manifest=(p/'SHA256SUMS').read_bytes();manifest=(p/'SHA256SUMS').read_text();blocks={}
  # Existing local bytes are still fully verified by transfer(). These hints
  # only avoid generating network-resume blocks which cannot be used for an
  # already present, unchanged manifest; they are not a source-integrity proof.
  already_local=known.get(p.name)==hashlib.sha256(raw_manifest).hexdigest()
  for line in manifest.splitlines():
   digest,name=line.split('  ',1);name=str(pathlib.PurePosixPath(name));file=p/name
   if not already_local and file.stat().st_size>32*1024**2:
    parts=[];offset=0
    with file.open('rb') as stream:
     while raw:=stream.read(4*1024**2):
      parts.append({'offset':offset,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});offset+=len(raw)
    blocks[name]=parts
  record={'name':p.name,'manifest':manifest,'blocks':blocks,'bytes':sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and 'restored' not in f.relative_to(p).parts),'protected_sets':sorted(protected)}
  if protection is not None:record['transition_protection']=protection
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


def known_local_manifests(root):
    """Cheap hints only. Transfer still reads every covered local file's SHA."""
    root = Path(root)
    from ops_windows_retention import safe_path
    from ops_runtime_contract import source_manifest_sha256
    result = {}
    for folder in sorted(root.iterdir()):
        if not NAME.fullmatch(folder.name):
            continue
        folder = safe_path(root, folder)
        manifest = safe_path(root, folder / 'SHA256SUMS')
        complete = safe_path(root, folder / 'COMPLETE')
        if complete.is_file() and manifest.is_file() and manifest.stat().st_size <= 4 * 1024**2:
            raw = manifest.read_bytes()
            result[folder.name] = (hashlib.sha256(raw).hexdigest() if (folder / 'SELFHOST_BACKUP.json').exists()
                                   else source_manifest_sha256(raw))
    return result


def read_inventory(command, *, hidden=None, log=None, run=None, sleep=None,
                   known=None, timeout=180):
    """At most three read-only attempts; never log private subprocess output.

    Each SSH invocation is bounded to 180 seconds (600 for explicit recovery).
    A timed-out remote read may survive SSH termination, so it is not retried.
    Authentication and host-key
    arguments remain identical; retries cannot authorize an untrusted peer.
    Invalid successful responses are validation failures, not transport retries.
    """
    run = subprocess.run if run is None else run
    sleep = time.sleep if sleep is None else sleep
    if type(timeout) is not int or timeout not in (180, 600):
        raise ValueError('bounded_inventory_timeout_required')
    if known is not None and (not isinstance(known, dict) or any(
            not isinstance(name, str) or not NAME.fullmatch(name)
            or not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha)
            for name, sha in known.items())):
        raise ValueError('invalid_local_manifest_hint')
    source = INVENTORY if not known else 'KNOWN_LOCAL_MANIFESTS = ' + repr(dict(sorted(known.items()))) + '\n' + INVENTORY
    for attempt in range(1, 4):
        try:
            result = run(command, input=source.encode(), capture_output=True,
                         timeout=timeout, **(hidden or {}))
        except subprocess.TimeoutExpired:
            if log:
                log('inventory_attempt_failed', attempt=attempt,
                    categories=['timeout'], remote_status='unknown_no_retry')
            raise RuntimeError('ssh_inventory_timeout_remote_status_unknown') from None
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


def writable_destination(root, path):
    """A mutable transfer file must not alias an existing immutable file."""
    from ops_windows_retention import safe_path
    path = safe_path(root, path)
    if path.is_file() and path.stat(follow_symlinks=False).st_nlink > 1:
        raise ValueError('multiply_linked_writable_destination')
    return path


def transfer_runtime(root, folder, record, download_many, free=shutil.disk_usage):
    """One SHA-addressed shared store; original covered dependencies stay unchanged."""
    if 'runtime' not in record:
        return None
    from ops_windows_retention import safe_path, verify_runtime
    runtime_file = folder/'RUNTIME_DEPENDENCIES.json'
    if json.loads(runtime_file.read_bytes()) != record['runtime']:
        raise ValueError('runtime_contract_changed')
    mapping = folder/'RELOCATED_RUNTIME.json'
    writing = writable_destination(root, mapping.with_name(mapping.name+'.writing'))
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
        temporary = writable_destination(root, target.with_name(target.name+'.downloading'))
        if matches(temporary, digest):
            writable_destination(root, temporary)
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
        for _, temporary in pending:
            writable_destination(root, temporary)
        download_many(None, pending)
        for source, temporary in pending:
            writable_destination(root, temporary)
            asset = next(a for a in assets if a['path'] == source)
            if not matches(temporary, asset['sha256']):
                # A failed resume cannot be retried as a complete asset.
                temporary.unlink(missing_ok=True)
                raise ValueError('checksum_mismatch')
            writable_destination(root, temporary)
            os.replace(temporary, shared/(asset['sha256']+'.archive'))
    writable_destination(root, writing)
    writing.write_text(json.dumps({'format':1,'assets':relocated}),encoding='utf-8')
    writable_destination(root, writing)
    os.replace(writing, mapping)
    return verify_runtime(root, folder)


def transfer(root, record, download, free=shutil.disk_usage, download_many=None, delta=None):
    from ops_windows_retention import canonical_root, safe_path
    root = canonical_root(root)
    name = record['name']
    if not NAME.fullmatch(name):
        raise ValueError('invalid_backup_name')
    files = entries(record['manifest'])
    native=('database.dump','roles.sql','source.json','archive-list.private.txt')
    if 'postgres/database.dump' in files and not all('postgres/'+n in files for n in (*native,'COMPLETE')):
        raise ValueError('incomplete_postgres_set')
    final = safe_path(root, root / name)
    if final.exists():
        for filename in (*files, 'SHA256SUMS', 'COMPLETE', 'OFFSITE_VERIFIED.json'):
            safe_path(root, final / filename)
        for filename in ('COMPLETE', 'OFFSITE_VERIFIED.json'):
            writable_destination(root, final / filename)
        if 'postgres/database.dump' in files and 'postgres/SHA256SUMS' not in files:
            writable_destination(root, final / 'postgres/SHA256SUMS')
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
            writable_destination(root, final/'postgres/SHA256SUMS').write_text(''.join(files['postgres/'+n]+'  '+n+'\n' for n in native), newline='\n')
        writable_destination(root, final/'COMPLETE').touch()
        return 'verified_existing'
    partial = safe_path(root, root / (name+'.partial'))
    partial.mkdir(exist_ok=True)
    # Reject all existing redirects before a download can write any bytes.
    for filename in (*files, 'SHA256SUMS', 'COMPLETE', 'OFFSITE_VERIFIED.json',
                     'postgres/SHA256SUMS'):
        safe_path(root, partial / filename)
    for filename in ('SHA256SUMS', 'COMPLETE', 'OFFSITE_VERIFIED.json'):
        writable_destination(root, partial / filename)
    if 'postgres/database.dump' in files and 'postgres/SHA256SUMS' not in files:
        writable_destination(root, partial / 'postgres/SHA256SUMS')
    for filename in files:
        target = partial / filename
        writable_destination(root, target.with_name(target.name + '.downloading'))
    if free(root).free < int(record['bytes']) + 2*1024**3:
        raise ValueError('insufficient_disk')
    pending=[]
    for filename, digest in files.items():
        target = safe_path(root, partial/filename)
        temporary = writable_destination(root, target.with_name(target.name+'.downloading'))
        target.parent.mkdir(parents=True, exist_ok=True)
        if matches(target, digest):
            continue
        if delta and delta(record,filename,target):
            if not matches(target,digest):raise ValueError('checksum_mismatch')
            # Only redundant staging bytes, after full-file verification. Never
            # remove a completed backup or the source used for matching blocks.
            temporary.unlink(missing_ok=True)
            continue
        if matches(temporary,digest):
            writable_destination(root, temporary)
            os.replace(temporary,target)
            continue
        pending.append((filename,temporary))
    if download_many and pending:
        for _, temporary in pending:
            writable_destination(root, temporary)
        download_many(name,pending)
    for filename,temporary in pending:
        digest=files[filename];target=safe_path(root, partial/filename)
        temporary=writable_destination(root, temporary)
        if not download_many:download(name, filename, temporary)
        writable_destination(root, temporary)
        if not matches(temporary, digest):
            quarantine=safe_path(root, root/'transfer-errors');quarantine.mkdir(exist_ok=True)
            temporary.rename(safe_path(root, quarantine/(name+'-'+str(time.time_ns())+'.bad-sha')))
            raise ValueError('checksum_mismatch')
        safe_path(root, target)
        writable_destination(root, temporary)
        os.replace(temporary, target)
    # Preserve the trusted SSH manifest byte for byte on Windows as well.
    # Translating LF to CRLF would change its hash used by external receipts.
    writable_destination(root, partial/'SHA256SUMS').write_text(record['manifest'], encoding='utf-8', newline='\n')
    writable_destination(root, partial/'OFFSITE_VERIFIED.json')
    result = verify(partial)
    transfer_runtime(root, partial, record, download_many, free)
    if 'postgres/database.dump' in files and 'postgres/SHA256SUMS' not in files:
        # Parent manifest covers native files; native manifest may be omitted by
        # outer find. The required four native files must all be present.
        writable_destination(root, partial/'postgres/SHA256SUMS').write_text(''.join(files['postgres/'+n]+'  '+n+'\n' for n in native), newline='\n')
        result['scope'] = 'one application/configuration/native PostgreSQL backup set'
    writable_destination(root, partial/'OFFSITE_VERIFIED.json').write_text(json.dumps(result,indent=2))
    writable_destination(root, partial/'COMPLETE').touch()
    safe_path(root, partial)
    safe_path(root, final)
    partial.rename(final)
    return 'downloaded_verified'


def preserve_transition_history(root, records):
    """After copying exact originals, install the trusted additive protection.

    A changed existing registry is refused; this is never implicit adoption or
    permission to rotate the original copies. Other Windows policy stays intact.
    """
    from ops_runtime_contract import TRANSITION_FILE, transition_protection, source_manifest_sha256
    from ops_windows_retention import safe_path
    values = [row['transition_protection'] for row in records if 'transition_protection' in row]
    if not values:
        return False
    value = values[0]
    if (any(item != value for item in values) or not isinstance(value, dict)
            or value.get('format') != 1 or value.get('kind') != 'selfhost_backup_transition'
            or not isinstance(value.get('sets'), list)):
        raise ValueError('invalid_transition_protection')
    target = safe_path(root, root / TRANSITION_FILE)
    writing = None if target.exists() else writable_destination(root, target.with_name(target.name + '.writing'))
    names = set()
    local_rows = []
    for row in value['sets']:
        if (not isinstance(row, dict) or not NAME.fullmatch(row.get('name', ''))
                or not re.fullmatch('[a-f0-9]{64}', row.get('manifest_sha256', ''))
                or ('local_manifest_sha256' in row and (not isinstance(row['local_manifest_sha256'], str) or not re.fullmatch(
                    '[a-f0-9]{64}', row.get('local_manifest_sha256', ''))
                    or row['local_manifest_sha256'] == row['manifest_sha256']))
                or row['name'] in names):
            raise ValueError('invalid_transition_protection')
        names.add(row['name'])
        folder = safe_path(root, root / row['name'])
        manifest = safe_path(root, folder / 'SHA256SUMS')
        if not safe_path(root, folder / 'COMPLETE').is_file() or not manifest.is_file():
            raise ValueError('transition_original_not_verified_locally')
        raw = manifest.read_bytes()
        local_sha = hashlib.sha256(raw).hexdigest()
        local_row = dict(row)
        if local_sha != row['manifest_sha256']:
            if source_manifest_sha256(raw) != row['manifest_sha256']:
                raise ValueError('transition_original_not_verified_locally')
            local_row['local_manifest_sha256'] = local_sha
        if 'local_manifest_sha256' in row and row['local_manifest_sha256'] != local_sha:
            raise ValueError('transition_original_not_verified_locally')
        # An unchanged manifest/hint alone does not prove its files. Verify all
        # covered bytes even on an idempotent adoption, without rewriting it.
        writable_destination(root, folder / 'OFFSITE_VERIFIED.json')
        verify(folder)
        local_rows.append(local_row)
    local_value = {**value, 'sets': local_rows}
    if target.exists():
        if json.loads(target.read_bytes()) != local_value:
            raise ValueError('existing_transition_protection_preserved')
    else:
        writable_destination(root, target)
        writable_destination(root, writing)
        with writing.open('w', encoding='utf-8') as output:
            json.dump(local_value, output, sort_keys=True); output.flush(); os.fsync(output.fileno())
        writable_destination(root, writing)
        writable_destination(root, target)
        os.replace(writing, target)
    transition_protection(root)
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default=r'D:\ArborScanBackups')
    parser.add_argument('--host',default='arborscan@31.57.170.88')
    parser.add_argument('--port',type=int,default=22)
    parser.add_argument('--retention-dry-run',action='store_true',
                        help='Verify transfer and save rotation plan without removing sets')
    parser.add_argument('--inventory-timeout',type=int,choices=(180,600),default=180,
                        help='Bounded initial recovery allowance; timed-out reads are never retried automatically')
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
                                    args.host,'python3 -'],hidden=hidden,log=log,
                                    known=known_local_manifests(root),timeout=args.inventory_timeout)
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
            preserve_transition_history(root, records)
            # Rotation is available only after a full native/runtime
            # replacement and SHA-addressed runtime readback. Old unadopted
            # manual/historical sets and live Compose dependencies remain pinned.
            latest = records[-1]
            runtime = latest.get('runtime', {})
            if runtime.get('selfhost') is True or set(runtime.get('services',[])) == {
                    '/arborscan-api','/arborscan-api-v4','/arborscan-quality-worker'}:
                from ops_windows_retention import plan, apply
                proposal = plan(root, latest['name'], latest.get('protected_sets', []))
                writable_destination(root, root/'retention-dry-run.private.json').write_text(json.dumps(proposal,indent=2))
                if args.retention_dry_run:
                    log('retention_dry_run',before=proposal['before'],candidates=len(proposal['delete']))
                else:
                    rotation = apply(root, proposal, latest.get('protected_sets', []), lock_held=True)
                    log('retention_complete', **rotation)
            log('success',complete_sets=len(records))
            return 0
        except Exception as error:
            # Never print subprocess output, connection details, or private paths.
            safe=str(error) if str(error) in ('ssh_inventory_failed','ssh_inventory_invalid_response',
                'ssh_inventory_timeout_remote_status_unknown','transfer_interrupted',
                'transfer_stalled','checksum_mismatch','insufficient_disk','existing_set_manifest_changed',
                'multiply_linked_writable_destination') else 'validation_or_local_error'
            log('failed',error_type=type(error).__name__,reason=safe)
            return 1


if __name__=='__main__':
    sys.exit(main())
