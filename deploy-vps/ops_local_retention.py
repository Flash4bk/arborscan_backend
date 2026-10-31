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
    if 'compact_policy = load_policy(root)' in source:
        return source
    delta = '            def delta(record,filename,target):'
    latest = '            latest = records[-1]'
    if source.count(delta) != 1 or source.count(latest) != 1:
        raise ValueError('unknown_installed_reader_layout')
    before = ('            from ops_local_retention import load_policy, select_records, plan as compact_plan, apply as compact_apply, write_plan as compact_write_plan\n'
              '            compact_policy = load_policy(root)\n'
              '            records = select_records(records, compact_policy)\n')
    block = ("            if compact_policy is not None:\n"
             "                proposal = compact_plan(root, compact_policy, selected=[r['name'] for r in records])\n"
             "                compact_write_plan(root,proposal)\n"
             "                if args.retention_dry_run:\n"
             "                    log('retention_dry_run',before=proposal['before'],candidates=len(proposal['delete']),retain=2)\n"
             "                else:\n"
             "                    log('retention_complete',**compact_apply(root,proposal,lock_held=True))\n"
             "                log('success',complete_sets=len(records),retain=2)\n"
             "                return 0\n")
    source = source.replace(delta, before + delta).replace(latest, block + latest)
    transition = '            preserve_transition_history(root, records)'
    if source.count(transition) > 1:
        raise ValueError('unknown_installed_reader_layout')
    if transition in source:
        source = source.replace(transition, '            if compact_policy is None:\n    ' + transition)
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
