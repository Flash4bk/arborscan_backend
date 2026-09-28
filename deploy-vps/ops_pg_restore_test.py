"""Restore into a new network-disabled disposable PostgreSQL container only.

Never mounts libpq credentials or application env. Restores roles/ACL/ownership.
Private errors are retained, not printed. The test DB is stopped in finally.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    os.umask(0o077)
    backup = Path(sys.argv[1]).resolve(strict=True)
    if not any(backup.is_relative_to(Path('/home/arborscan')/p) for p in ('pg-backups','ops-backups')) or not (backup/'COMPLETE').exists():
        raise ValueError('Verified native backup required')
    for line in (backup/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (backup/name).resolve(strict=True)
        if not path.is_relative_to(backup):
            raise ValueError('Unsafe manifest')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Checksum mismatch')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = 'as14-pg-restore-' + stamp.lower()
    result_dir = backup / ('restore-' + stamp)
    result_dir.mkdir(mode=0o700)
    image = subprocess.check_output(['docker','image','inspect','supabase/postgres:17.6.1.063','--format','{{.Id}}'], text=True).strip()
    started = time.monotonic()
    def run(args, data=None, required=True):
        with (result_dir/'diagnostics.private.log').open('ab') as log:
            result = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=log, timeout=600)
        if required and result.returncode:
            raise RuntimeError('Restore command failed')
        return result
    base = ['docker','exec','-i',name]
    sql = base + ['psql','-h','/tmp','-U','supabase_admin','-d','postgres','-v','ON_ERROR_STOP=1','-At']
    try:
        run(['docker','run','-d','--name',name,'--network','none','--user','postgres',
             '--memory','2g','--cpus','1','--pids-limit','128',
             '--tmpfs','/tmp:rw,size=2147483648,mode=1777',
             '--entrypoint','/bin/bash',image,'-c',
             'initdb -D /tmp/as14-data -U supabase_admin --auth=trust >/tmp/init.log 2>&1 && exec postgres -D /tmp/as14-data -k /tmp -c listen_addresses= -c shared_preload_libraries='])
        for _ in range(30):
            if run(base+['pg_isready','-h','/tmp','-U','supabase_admin'], required=False).returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Test database not ready')
        # Only backup data enters the isolated container, not production credentials.
        roles = (backup/'roles.sql').read_text()
        grants = [line for line in roles.splitlines() if line.startswith('GRANT ')]
        if any(not line.endswith(' GRANTED BY supabase_admin;') for line in grants):
            raise RuntimeError('Unexpected grantor: review isolated restore procedure')
        # PostgreSQL requires these grants to execute as their recorded grantor.
        # Preserve membership options and grantor; do not drop ACLs or promote roles.
        definitions = '\n'.join(line for line in roles.splitlines()
                                if not line.startswith('GRANT ') and line != 'CREATE ROLE supabase_admin;')
        run(sql, definitions.encode())
        run(sql, ('SET ROLE supabase_admin;\n'+'\n'.join(grants)+'\nRESET ROLE;').encode())
        run(base+['pg_restore','-h','/tmp','-U','supabase_admin','-d','postgres',
                  '--exit-on-error','--single-transaction'], (backup/'database.dump').read_bytes())
        scripts = Path(__file__).parent
        run(sql, (scripts/'as14_legacy_access.sql').read_bytes())
        run(sql, b"SET as14.restore_test='enabled';\n" + (scripts/'as14_restored_workflow.sql').read_bytes())
        linked = 0
        if len(sys.argv) > 2:
            files = Path(sys.argv[2]).resolve(strict=True)
            if not files.is_relative_to(Path('/home/arborscan/ops-backups')):
                raise ValueError('Expected private application restore path')
            for table, fields in (
                ('report_versions',('owner_id','version_id','parent_id','payload_sha256','image_sha256','correction_id')),
                ('contour_revisions',('owner_id','correction_id','parent_id','image_sha256','status','decisions'))):
                actual = json.loads(run(sql, f"select coalesce(json_agg(t),'[]') from public.{table} t;".encode()).stdout)
                expected = json.loads((files/'tables'/f'{table}.json').read_text())
                def normalized(rows):
                    return sorted(json.dumps({f:r[f] for f in fields},sort_keys=True) for r in rows)
                if normalized(actual) != normalized(expected):
                    raise RuntimeError('DB and application snapshot differ; linked restore not verified')
                linked += len(actual)
        summary = run(sql, b"select json_build_object('tables',(select count(*) from pg_class c join pg_namespace n on n.oid=c.relnamespace where c.relkind='r' and n.nspname not like 'pg_%' and n.nspname<>'information_schema'),'contour_version',public.contour_workflow_version(),'history_version',public.server_history_version(),'quality_version',public.model_quality_version(),'reports',(select count(*) from public.report_versions),'contours',(select count(*) from public.contour_revisions));").stdout
        result = {'container':name,'image':image,'network':'none','production_credentials_mounted':False,
                  'seconds':round(time.monotonic()-started,3),'restored':json.loads(summary),
                  'acl_and_workflow_tests':'passed','rows_matched_to_file_snapshot':linked}
        (result_dir/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
    finally:
        subprocess.run(['docker','stop',name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Isolated restore not completed; private diagnostics retained')
        raise SystemExit(1)
