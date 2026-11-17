"""Explicit owner-selected Windows latest-two policy; absent policy keeps legacy14.

Only local timestamp backup directories are disposable. Shared runtime, tools,
signing material and recovery fixtures never enter this module's delete list.
The existing installed verifier remains authoritative for native/runtime data.
"""
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

POLICY_FILE = 'LOCAL_LATEST_TWO_POLICY.private.json'
PREPARED_POLICY_FILE = 'LOCAL_LATEST_TWO_POLICY.prepared.private.json'
NAME = re.compile(r'\d{8}T\d{6}Z')
SHA = re.compile(r'[a-f0-9]{64}')

# Inserted after the installed inventory's complete live/protection discovery.
# Eligibility is small metadata only; the original selected payload/runtime
# hashing and the installed local verifiers remain mandatory afterwards.
_REMOTE_SELECTION = r'''
def compact_file(folder,name,limit=None):
 path=folder
 for part in pathlib.PurePosixPath(name).parts:
  path=path/part
  if path.is_symlink():raise ValueError('unsafe_latest_two_metadata')
 if not path.is_file() or (limit is not None and path.stat().st_size>limit):raise ValueError('invalid_latest_two_metadata')
 return path
def compact_read(folder,name,limit):
 with compact_file(folder,name,limit).open('rb') as stream:raw=stream.read(limit+1)
 if len(raw)>limit:raise ValueError('invalid_latest_two_metadata')
 return raw
def compact_entries(raw):
 entries={}
 for line in raw.decode('utf-8').splitlines():
  expected,name=line.split('  ',1)
  parsed=pathlib.PurePosixPath(name)
  if (not re.fullmatch('[a-f0-9]{64}',expected) or name in entries or str(parsed)!=name or parsed.is_absolute() or '..' in parsed.parts or name=='.' or not re.fullmatch('[A-Za-z0-9_./-]+',name)):raise ValueError('invalid_latest_two_manifest')
  entries[name]=expected
 return entries
def compact_metadata(folder,name,entries,limit):
 raw=compact_read(folder,name,limit)
 if hashlib.sha256(raw).hexdigest()!=entries.get(name):raise ValueError('latest_two_metadata_checksum_mismatch')
 return raw
for ancestor in [root,*root.parents]:
 if ancestor.is_symlink():raise ValueError('unsafe_latest_two_metadata')
compact_selected=[]
compact_native={'database.dump','roles.sql','source.json','archive-list.private.txt'}
compact_required={'application.tar','local-files.tar','arborscan.env.private','containers.private.json'}
for candidate in sorted(root.iterdir(),reverse=True):
 if not re.fullmatch(r'\d{8}T\d{6}Z',candidate.name):continue
 if candidate.is_symlink():raise ValueError('unsafe_latest_two_metadata')
 complete=candidate/'COMPLETE'
 if complete.is_symlink():raise ValueError('unsafe_latest_two_metadata')
 if not complete.exists():continue
 compact_file(candidate,'COMPLETE')
 covered=compact_entries(compact_read(candidate,'SHA256SUMS',4*1024**2))
 if not compact_required<=covered.keys():raise ValueError('invalid_latest_two_manifest')
 for name in covered:compact_file(candidate,name)
 modern=(any(name=='postgres' or name.startswith('postgres/') or name in ('RUNTIME_DEPENDENCIES.json','SELFHOST_BACKUP.json') for name in covered)
         or any((candidate/name).exists() or (candidate/name).is_symlink() for name in ('postgres','RUNTIME_DEPENDENCIES.json','SELFHOST_BACKUP.json')))
 if not modern:continue
 required={'postgres/'+name for name in compact_native|{'COMPLETE','SHA256SUMS'}}|{'RUNTIME_DEPENDENCIES.json'}
 if not required<=covered.keys():raise ValueError('incomplete_latest_two_native_runtime')
 native=compact_entries(compact_metadata(candidate,'postgres/SHA256SUMS',covered,1024**2))
 if native.keys()!=compact_native or any(covered['postgres/'+name]!=sha for name,sha in native.items()):raise ValueError('invalid_latest_two_native_manifest')
 runtime=json.loads(compact_metadata(candidate,'RUNTIME_DEPENDENCIES.json',covered,2*1024**2))
 if not isinstance(runtime,dict) or type(runtime.get('format')) is not int or runtime.get('format')!=1 or not isinstance(runtime.get('images'),dict) or not runtime['images']:raise ValueError('invalid_latest_two_runtime')
 for image,assets in runtime['images'].items():
  if not re.fullmatch('sha256:[a-f0-9]{64}',image) or not isinstance(assets,list) or not assets:raise ValueError('invalid_latest_two_runtime')
  for asset in assets:
   if not isinstance(asset,dict) or not isinstance(asset.get('path'),str) or not isinstance(asset.get('sha256'),str) or not re.fullmatch('[a-f0-9]{64}',asset['sha256']):raise ValueError('invalid_latest_two_runtime')
   path=pathlib.Path(asset['path'])
   if not path.is_absolute() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():raise ValueError('unsafe_latest_two_runtime')
 if 'SELFHOST_BACKUP.json' in covered or (candidate/'SELFHOST_BACKUP.json').exists():
  if runtime.get('selfhost') is not True:raise ValueError('invalid_latest_two_runtime')
 if runtime.get('selfhost') is True:
  marker=json.loads(compact_metadata(candidate,'SELFHOST_BACKUP.json',covered,2*1024**2))
  if not isinstance(marker,dict) or type(marker.get('format')) is not int or marker.get('format')!=1 or marker.get('kind')!='local_selfhost_backup':raise ValueError('invalid_latest_two_selfhost_marker')
 compact_selected.append(candidate)
 if len(compact_selected)==2:break
if len(compact_selected)!=2:raise ValueError('two_native_runtime_backups_required')
compact_selected.reverse()
for p in compact_selected:
'''


def inventory_source(source, policy):
    """Opt in before heavy inventory; absent policy preserves exact source bytes."""
    if policy is None:
        return source
    loop = 'for p in sorted(root.iterdir()):\n'
    if (source.count(loop) != 1 or source.count('runtime_checked={}') != 1
            or source.count('print(json.dumps(result))') != 1
            or not source.index('runtime_checked={}') < source.index(loop) < source.index('print(json.dumps(result))')):
        raise ValueError('unknown_installed_inventory_layout')
    return source.replace(loop, _REMOTE_SELECTION.lstrip('\n'))


def write_plan(root, proposal):
    ret = _retention()
    root = ret.canonical_root(root)
    target = ret.safe_path(root, root / 'retention-dry-run.private.json')
    if target.exists() and target.stat().st_nlink != 1:
        raise ValueError('multiply_linked_writable_destination')
    temporary = ret.safe_path(root, root / ('.retention-plan-' + uuid.uuid4().hex))
    with temporary.open('x', encoding='utf-8') as output:
        json.dump(proposal, output, indent=2)
        output.flush(); os.fsync(output.fileno())
    ret.safe_path(root, target)
    os.replace(temporary, target)


def adapt_reader_source(source):
    """Add only opt-in hooks to the exact installed reader, preserving its fixes."""
    delta = '            def delta(record,filename,target):'
    latest = '            latest = records[-1]'
    before = ('            from ops_local_retention import load_policy, select_records, plan as compact_plan, apply as compact_apply, write_plan as compact_write_plan\n'
              '            compact_policy = load_policy(root)\n'
              '            records = select_records(records, compact_policy)\n')
    early = ('            from ops_local_retention import load_policy, select_records, inventory_source as compact_inventory_source, plan as compact_plan, apply as compact_apply, write_plan as compact_write_plan\n'
             '            compact_policy = load_policy(root)\n'
             '            # local_latest_two_inventory_v2\n')
    option = 'inventory_source=compact_inventory_source(INVENTORY, compact_policy)'
    signature = ('def read_inventory(command, *, hidden=None, log=None, run=None, sleep=None,\n'
                 '                   known=None, timeout=180):')
    new_signature = signature.replace('timeout=180):', 'timeout=180, inventory_source=None):')
    source_line = "    source = INVENTORY if not known else 'KNOWN_LOCAL_MANIFESTS = ' + repr(dict(sorted(known.items()))) + '\\n' + INVENTORY"
    new_source_line = ("    inventory = INVENTORY if inventory_source is None else inventory_source\n"
                       "    source = inventory if not known else 'KNOWN_LOCAL_MANIFESTS = ' + repr(dict(sorted(known.items()))) + '\\n' + inventory")
    legacy_signature = 'def read_inventory(command, *, hidden=None, log=None, run=None, sleep=None):'
    legacy_new_signature = legacy_signature.replace('sleep=None):', 'sleep=None, inventory_source=None):')
    legacy_input = 'input=INVENTORY.encode(), capture_output=True,'
    legacy_new_input = 'input=(INVENTORY if inventory_source is None else inventory_source).encode(), capture_output=True,'
    block = ("            if compact_policy is not None:\n"
             "                proposal = compact_plan(root, compact_policy, selected=[r['name'] for r in records])\n"
             "                compact_write_plan(root,proposal)\n"
             "                if args.retention_dry_run:\n"
             "                    log('retention_dry_run',before=proposal['before'],candidates=len(proposal['delete']),retain=2)\n"
             "                else:\n"
             "                    log('retention_complete',**compact_apply(root,proposal,lock_held=True))\n"
             "                log('success',complete_sets=len(records),retain=2)\n"
             "                return 0\n")
    inventory_call = '            records=read_inventory('
    selected = '            records = select_records(records, compact_policy)\n'
    if 'local_latest_two_inventory_v2' in source:
        valid_transport = ((source.count(new_signature) == 1 and source.count(new_source_line) == 1)
                           or (source.count(legacy_new_signature) == 1 and source.count(legacy_new_input) == 1))
        if (source.count(early) != 1 or source.count(option) != 1 or not valid_transport
                or source.count('local_latest_two_inventory_v2') != 1
                or source.count('def read_inventory(') != 1 or source.count(inventory_call) != 1
                or source.count(selected) != 1 or source.count(block) != 1
                or source.count('compact_policy = load_policy(root)') != 1
                or not source.index(early) < source.index(inventory_call) < source.index(selected)):
            raise ValueError('unknown_installed_reader_layout')
        return source
    if source.count(delta) != 1 or source.count(latest) != 1:
        raise ValueError('unknown_installed_reader_layout')
    if 'compact_policy = load_policy(root)' in source:
        if source.count(before) != 1 or source.count(block) != 1:
            raise ValueError('unknown_installed_reader_layout')
    else:
        source = source.replace(delta, before + delta).replace(latest, block + latest)
        transition = '            preserve_transition_history(root, records)'
        if source.count(transition) > 1:
            raise ValueError('unknown_installed_reader_layout')
        if transition in source:
            source = source.replace(transition, '            if compact_policy is None:\n    ' + transition)
    if source.count(inventory_call) != 1 or source.count('def read_inventory(') != 1:
        raise ValueError('unknown_installed_reader_layout')
    source = source.replace(before, selected)
    source = source.replace(inventory_call, early + inventory_call)
    if source.count(signature) == 1 and source.count(source_line) == 1:
        argument = 'known=known_local_manifests(root),timeout=args.inventory_timeout)'
        if source.count(argument) != 1:
            raise ValueError('unknown_installed_reader_layout')
        source = source.replace(signature, new_signature).replace(source_line, new_source_line)
        source = source.replace(argument, argument[:-1] + ',' + option + ')')
    elif source.count(legacy_signature) == 1 and source.count(legacy_input) == 1:
        argument = "args.host,'python3 -'],hidden=hidden,log=log)"
        if source.count(argument) != 1:
            raise ValueError('unknown_installed_reader_layout')
        source = source.replace(legacy_signature, legacy_new_signature).replace(legacy_input, legacy_new_input)
        source = source.replace(argument, argument[:-1] + ',' + option + ')')
    else:
        raise ValueError('unknown_installed_reader_layout')
    return source


def _retention():
    import ops_windows_retention
    return ops_windows_retention


def load_policy(root, filename=POLICY_FILE):
    ret = _retention()
    root = ret.canonical_root(root)
    if filename not in (POLICY_FILE, PREPARED_POLICY_FILE):
        raise ValueError('invalid_local_policy_filename')
    path = ret.safe_path(root, root / filename)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('format') != 1 or value.get('kind') != 'owner_selected_local_latest_two'
            or value.get('retain') != 2 or value.get('authorization') != 'explicit_owner_request'
            or value.get('root') != str(root) or not value.get('authorized_at')
            or not isinstance(value.get('released_sets'), list)):
        raise ValueError('invalid_local_latest_two_policy')
    seen = set()
    for row in value['released_sets']:
        if (not isinstance(row, dict) or not NAME.fullmatch(row.get('name', ''))
                or not SHA.fullmatch(row.get('manifest_sha256', ''))
                or row['name'] in seen or not isinstance(row.get('pins'), list)
                or any(p not in ('PINNED', 'DO_NOT_ROTATE', 'ROLLBACK_PIN') for p in row['pins'])):
            raise ValueError('invalid_local_release_ledger')
        seen.add(row['name'])
    return value


def select_records(records, policy):
    """Select before transfer so remote historical14 cannot reappear locally."""
    if policy is None:
        return records
    names = set()
    candidates = []
    from ops_pull_backups import entries
    required = {'postgres/' + n for n in ('database.dump', 'roles.sql', 'source.json',
                'archive-list.private.txt', 'COMPLETE', 'SHA256SUMS')}
    for row in records:
        name = row.get('name', '')
        if not NAME.fullmatch(name) or name in names:
            raise ValueError('invalid_local_source_inventory')
        names.add(name)
        manifest = entries(row['manifest'])
        if required <= manifest.keys() and row.get('runtime', {}).get('images'):
            candidates.append(row)
    selected = sorted(candidates, key=lambda row: row['name'])[-2:]
    if len(selected) != 2:
        raise ValueError('two_native_runtime_backups_required')
    return selected


def _no_reparse_tree(folder):
    ret = _retention()
    for current, dirs, files in os.walk(folder, followlinks=False):
        for name in [*dirs, *files]:
            ret.safe_path(folder, Path(current) / name)


def plan(root, policy=None, selected=None, policy_file=POLICY_FILE):
    ret = _retention()
    root = ret.canonical_root(root)
    current = load_policy(root, policy_file)
    if current is None or (policy is not None and policy != current):
        raise ValueError('local_policy_missing_or_changed')
    policy = current
    complete = [p for p in sorted(root.iterdir()) if NAME.fullmatch(p.name)
                and p.is_dir() and (p / 'COMPLETE').is_file()]
    keep = sorted(selected) if selected is not None else [p.name for p in complete[-2:]]
    if (len(keep) != 2 or len(set(keep)) != 2
            or any(not NAME.fullmatch(n) for n in keep)
            or any(n not in {p.name for p in complete} for n in keep)):
        raise ValueError('two_complete_local_backups_required')
    # If a more recent local set exists, do not prune it using stale source data.
    if keep != [p.name for p in complete[-2:]]:
        raise ValueError('local_source_inventory_is_stale')
    kept = []
    runtime = {}
    for name in keep:
        folder = ret.safe_path(root, root / name)
        verified = ret.verify_replacement(root, name)
        kept.append({'name': name, 'manifest_sha256': ret.digest(folder / 'SHA256SUMS'),
                     'native_runtime_verified': verified})
        relocation = json.loads((folder / 'RELOCATED_RUNTIME.json').read_text(encoding='utf-8'))
        for asset in relocation['assets']:
            target = ret.safe_path(root, asset['restored_path'])
            relative = target.relative_to(root)
            if relative.parts[0] not in keep and relative.parts[0] != 'runtime-assets':
                raise ValueError('kept_backup_depends_on_disposable_directory')
            actual = ret.digest(target)
            if actual != asset['sha256']:
                raise ValueError('runtime_archive_checksum_mismatch')
            runtime[str(relative)] = actual
    released = {r['name']: r for r in policy['released_sets']}
    delete = []
    for folder in complete:
        ret.safe_path(root, folder)
        if folder.name in keep:
            continue
        _no_reparse_tree(folder)
        manifest_sha = ret.digest(folder / 'SHA256SUMS')
        pins = sorted(n for n in ('PINNED', 'DO_NOT_ROTATE', 'ROLLBACK_PIN') if (folder / n).exists())
        release = released.get(folder.name)
        if release:
            if release['manifest_sha256'] != manifest_sha or sorted(release['pins']) != pins:
                raise ValueError('released_backup_changed')
        else:
            if pins:
                raise ValueError('new_pinned_backup_is_not_released')
            covered = ret.verify_data(folder)
            if 'AUTO_BACKUP.json' not in covered:
                raise ValueError('unreleased_historical_backup')
            marker = json.loads((folder / 'AUTO_BACKUP.json').read_text(encoding='utf-8'))
            if (marker.get('format') != 1 or marker.get('kind') != 'scheduled_backup'
                    or marker.get('name') != folder.name):
                raise ValueError('unreleased_historical_backup')
        delete.append({'name': folder.name, 'manifest_sha256': manifest_sha, 'pins': pins})
    return {'format': 1, 'kind': 'local_latest_two_plan', 'root': str(root),
            'policy_sha256': ret.digest(root / policy_file), 'retain': 2,
            'prepared_only': policy_file != POLICY_FILE,
            'before': len(complete), 'keep': kept, 'delete': delete,
            'shared_runtime': runtime, 'postgres_restore_performed': False}


def _powershell_remove(root, rows, *, keep, policy_sha256):
    """Native single-shell removal with all absolute paths revalidated in PowerShell."""
    if not rows:
        return []
    if os.name != 'nt':
        raise ValueError('native_windows_removal_required')
    payload = base64.b64encode(json.dumps({'root': str(root), 'rows': rows,
        'keep':keep,'policy_sha256':policy_sha256,'policy_file':POLICY_FILE}).encode()).decode()
    script = r'''
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$data=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD')) | ConvertFrom-Json
$root=[IO.Path]::GetFullPath($data.root).TrimEnd('\')
function Assert-Root {
  $ancestor=Get-Item -LiteralPath $root -Force
  while($ancestor) {
    if($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Unsafe backup root'}
    $ancestor=$ancestor.Parent
  }
}
function Get-Sha($path) {
  $item=Get-Item -LiteralPath $path -Force
  if($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Unsafe metadata path'}
  $stream=[IO.File]::OpenRead($path)
  $sha=[Security.Cryptography.SHA256]::Create()
  try {return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant()}
  finally {$stream.Dispose();$sha.Dispose()}
}
function Assert-Row($row) {
  if($row.name -notmatch '^\d{8}T\d{6}Z$') {throw 'Unsafe backup name'}
  $path=[IO.Path]::GetFullPath((Join-Path $root $row.name))
  if([IO.Path]::GetDirectoryName($path) -ne $root) {throw 'Unsafe backup target'}
  $item=Get-Item -LiteralPath $path -Force
  if(-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {throw 'Unsafe backup target'}
  foreach($child in Get-ChildItem -LiteralPath $path -Force -Recurse) {
    if($child.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Unsafe child target'}
  }
  $actual=Get-Sha (Join-Path $path 'SHA256SUMS')
  if($actual -ne $row.manifest_sha256) {throw 'Backup changed'}
  $pins=@('PINNED','DO_NOT_ROTATE','ROLLBACK_PIN' | Where-Object {Test-Path -LiteralPath (Join-Path $path $_)} | Sort-Object)
  if(($pins -join ',') -ne (@($row.pins | Sort-Object) -join ',')) {throw 'Backup pins changed'}
  return $path
}
Assert-Root
foreach($row in $data.rows) {$null=Assert-Row $row}
foreach($row in $data.rows) {
  Assert-Root
  if((Get-Sha (Join-Path $root $data.policy_file)) -ne $data.policy_sha256) {throw 'Local policy changed'}
  foreach($kept in $data.keep) {
    $path=[IO.Path]::GetFullPath((Join-Path $root $kept.name))
    if([IO.Path]::GetDirectoryName($path) -ne $root -or -not (Test-Path -LiteralPath (Join-Path $path 'COMPLETE'))) {throw 'Kept backup missing'}
    if((Get-Sha (Join-Path $path 'SHA256SUMS')) -ne $kept.manifest_sha256) {throw 'Kept backup changed'}
  }
  $path=Assert-Row $row
  Remove-Item -LiteralPath $path -Recurse -Force
  Write-Output $row.name
}
'''.replace('PAYLOAD', payload)
    encoded = base64.b64encode(script.encode('utf-16-le')).decode()
    completed = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                               '-EncodedCommand', encoded], capture_output=True,
                              text=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:
        raise ValueError('native_windows_removal_failed')
    removed = completed.stdout.split()
    if removed != [r['name'] for r in rows]:
        raise ValueError('native_windows_removal_receipt_mismatch')
    return removed


def apply(root, proposal, lock_held=False):
    if proposal.get('prepared_only'):
        raise ValueError('prepared_local_policy_cannot_delete')
    ret = _retention()
    root = ret.canonical_root(root)
    if not lock_held:
        with ret.locked(root):
            return apply(root, proposal, lock_held=True)
    current = plan(root, selected=[r['name'] for r in proposal['keep']])
    if current != proposal:
        raise ValueError('local_retention_plan_changed')
    deleted = _powershell_remove(root, proposal['delete'],keep=proposal['keep'],
                                 policy_sha256=proposal['policy_sha256'])
    final = plan(root)
    if final['delete'] or final['before'] != 2:
        raise ValueError('local_retention_postcheck_failed')
    return {'deleted': deleted, 'retained_verified_sets': 2,
            'shared_runtime_archives_preserved': len(final['shared_runtime'])}
