"""Actual isolated SQL/raw-file restore of a verified local self-host backup.

No API/worker starts. The fresh PostgreSQL container has no network/credentials,
stops in finally, and never deletes copied sessions/jobs. New raw volumes are
retained as evidence. Production volumes/DB are not accepted as restore targets.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tarfile
import time

from ops_selfhost_backup import (BACKUP_ROOT, PRIVATE_HOME, IMAGE, member_name,
    safe_path, verify_set, canonical, normalized_inventory, STORAGE_SQL)
from ops_selfhost_stack import role_restore_sql

VOLUME_CHECK = r'''
import base64,hashlib,json,os,pathlib,sys
sys.path.insert(0,'/tools');import ops_selfhost_backup as b
archive=pathlib.Path('/snapshot')/sys.argv[1];target=pathlib.Path('/restore-volume')
result=b.restore_volume_archive(archive,target)
expected=b.verify_file_tar(archive);meta=b.verify_filesystem_metadata(archive,expected)
for group in ('files','directories'):
 for name,row in meta[group].items():
  path=target if name=='.' else target.joinpath(*pathlib.PurePosixPath(name).parts);stat=path.stat()
  if stat.st_uid!=row['uid'] or stat.st_gid!=row['gid'] or stat.st_mode&0o777!=row['mode']:raise RuntimeError('Private metadata mismatch')
  if stat.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('Private timestamp mismatch')
  for attribute,encoded in row['xattrs'].items():
   if os.getxattr(path,attribute,follow_symlinks=False)!=base64.b64decode(encoded,validate=True):raise RuntimeError('Private xattr mismatch')
for name in meta['files']:
 if b.digest(target.joinpath(*pathlib.PurePosixPath(name).parts))!=expected[name]:raise RuntimeError('Private restored byte mismatch')
print(json.dumps(result))
'''


def normalized_rows(rows):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('invalid_restored_table')
    return sorted(canonical(row) for row in rows)


def check(backup, evidence_root, *, run=subprocess.run, sleep=time.sleep):
    backup, evidence_root = safe_path(backup), safe_path(evidence_root)
    if backup.parent != BACKUP_ROOT or not re.fullmatch(r'\d{8}T\d{6}Z', backup.name) or not (backup / 'COMPLETE').is_file():
        raise ValueError('verified_local_backup_required')
    if (evidence_root.parent.parent != PRIVATE_HOME or evidence_root.name != 'backup-restore-checks'
            or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', evidence_root.parent.name)):
        raise ValueError('isolated_evidence_root_required')
    verify_set(backup)
    rows = json.loads((backup / 'containers.private.json').read_bytes())
    roles = {r['Config']['Labels']['com.docker.compose.service']: r for r in rows}
    db_image, api_image = roles['db']['Image'], roles['api']['Image']
    if not IMAGE.fullmatch(db_image) or not IMAGE.fullmatch(api_image):
        raise ValueError('exact_restore_images_required')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(6)
    evidence = evidence_root / stamp; evidence.mkdir(mode=0o700, parents=True)
    os.umask(0o077)
    log = evidence / 'diagnostics.private.log'
    name = 'as14-selfhost-restore-' + stamp.lower()
    started = False
    def command(args, *, data=None, stdin=None, required=True, timeout=1800):
        with log.open('ab') as error:
            reply = run(args, input=data, stdin=stdin, stdout=subprocess.PIPE, stderr=error, timeout=timeout)
        if required and reply.returncode:
            raise RuntimeError('isolated_restore_command_failed')
        return reply
    base = ['docker', 'exec', '-i', name]
    sql = base + ['psql', '-h', '/tmp', '-U', 'supabase_admin', '-d', 'postgres',
                  '-v', 'ON_ERROR_STOP=1', '-X', '-qAt']
    def query(text):
        return json.loads(command(sql, data=text.encode()).stdout)
    try:
        command(['docker', 'run', '-d', '--name', name, '--network', 'none', '--user', 'postgres',
            '--memory', '2g', '--cpus', '1', '--pids-limit', '128',
            '--tmpfs', '/tmp:rw,size=2147483648,mode=1777', '--entrypoint', '/bin/bash', db_image,
            '-c', 'initdb -D /tmp/as14-restore -U supabase_admin --auth=trust >/tmp/init.log 2>&1 && '
                  'exec postgres -D /tmp/as14-restore -k /tmp -c listen_addresses= -c shared_preload_libraries='])
        started = True
        for _ in range(60):
            if command(base + ['pg_isready', '-h', '/tmp', '-U', 'supabase_admin'], required=False).returncode == 0:
                break
            sleep(1)
        else:
            raise RuntimeError('isolated_restore_db_not_ready')
        command(sql, data=role_restore_sql((backup / 'postgres/roles.sql').read_text()).encode())
        with (backup / 'postgres/database.dump').open('rb') as source:
            command(base + ['pg_restore', '-h', '/tmp', '-U', 'supabase_admin', '-d', 'postgres',
                           '--exit-on-error', '--single-transaction'], stdin=source)
        matched = {}
        with tarfile.open(backup / 'application.tar') as application:
            for member in application:
                if not member.name.startswith('tables/') or not member.name.endswith('.json'):
                    continue
                member_name(member.name)
                table = Path(member.name).stem
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table):
                    raise ValueError('invalid_restored_table_name')
                expected = json.load(application.extractfile(member))
                actual = query('SET TIME ZONE \'UTC\'; SELECT coalesce(json_agg(t), \'[]\'::json) FROM public."' + table + '" t;')
                if normalized_rows(actual) != normalized_rows(expected):
                    raise RuntimeError('restored_public_rows_mismatch')
                matched[table] = len(actual)
        if not matched:
            raise ValueError('restored_tables_missing')
        marker = json.loads((backup / 'SELFHOST_BACKUP.json').read_bytes())
        inventory = normalized_inventory(query(STORAGE_SQL))
        if hashlib.sha256(canonical(inventory)).hexdigest() != marker['storage_inventory_sha256']:
            raise RuntimeError('restored_storage_inventory_mismatch')
        versions = query('SELECT json_build_array(public.contour_workflow_version(), public.server_history_version(), public.model_quality_version());')
        if versions != [1, 1, 1]:
            raise RuntimeError('restored_migrations_mismatch')
        volumes = {}
        for archive in ('storage-files.tar', 'runtime-files.tar'):
            volume = 'as14-selfhost-' + archive.removesuffix('.tar') + '-' + stamp.lower()
            # Name is fresh, generated by this invocation; no volume is reused.
            if command(['docker', 'volume', 'inspect', volume], required=False).returncode == 0:
                raise RuntimeError('existing_restore_volume_preserved')
            command(['docker', 'volume', 'create', volume])
            tools = str(Path(__file__).parent.resolve())
            reply = command(['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only',
                '--user', '0:0', '--mount', 'type=bind,src=' + tools + ',dst=/tools,readonly',
                '--mount', 'type=bind,src=' + str(backup) + ',dst=/snapshot,readonly',
                '--mount', 'type=volume,src=' + volume + ',dst=/restore-volume',
                '--entrypoint', 'python', api_image, '-c', VOLUME_CHECK, archive])
            volumes[archive] = {**json.loads(reply.stdout), 'volume': volume}
        result = {'event': 'isolated_selfhost_restore_verified', 'backup': backup.name,
            'restored_backup': str(backup), 'restore_verified': True,
            'backup_manifest_sha256': hashlib.sha256((backup / 'SHA256SUMS').read_bytes()).hexdigest(),
            'db_image': db_image, 'network': 'none', 'production_credentials_mounted': False,
            'native_sql_restore': True, 'public_tables_matched': len(matched),
            'auth_sessions_rows_matched': matched.get('auth_sessions', 0),
            'quality_jobs_rows_matched': matched.get('ml_jobs', 0),
            'storage_objects_matched': len(inventory['objects']), 'migration_versions': versions,
            'roles_owner_acl_restored_by_pg_restore': True, 'raw_volumes': volumes,
            'worker_started': False, 'training_started': False}
        (evidence / 'result.json').write_bytes(canonical(result))
        print(json.dumps(result))
        return result
    finally:
        if started:
            command(['docker', 'stop', name], required=False, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', required=True)
    parser.add_argument('--evidence-root', required=True)
    args = parser.parse_args()
    try:
        check(args.backup, args.evidence_root)
    except Exception:
        print(json.dumps({'event': 'isolated_selfhost_restore_failed',
                          'production_data_modified': False}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
