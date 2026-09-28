"""Read-only native PostgreSQL dump using private libpq files and pinned client.

No production credentials are copied into the backup. All diagnostics private.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import fcntl

IMAGE = 'postgres@sha256:d74eeac9a635390a49bc21bd49fccd973de707e2a53a76ac49b552b8712ec46f'
HOME = Path('/home/arborscan')
CONNECTION = 'service=arborscan_backup sslmode=verify-full sslrootcert=/home/arborscan/supabase-ca.crt'


def client(tool, args, output_dir):
    command = ['docker', 'run', '--rm', '--network', 'host', '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', f'{output_dir}:{output_dir}', '-w', str(output_dir), '-e', 'PGCONNECT_TIMEOUT=20']
    command += ['-v', '/home/arborscan/supabase-ca.crt:/home/arborscan/supabase-ca.crt:ro']
    for name, variable in (('.pg_service.conf', 'PGSERVICEFILE'), ('.pgpass', 'PGPASSFILE')):
        path = HOME / name
        if path.stat().st_mode & 0o077:
            raise RuntimeError('libpq file permissions must be 600')
        command += ['-v', f'{path}:{path}:ro', '-e', f'{variable}={path}']
    return command + [IMAGE, tool] + args


def main():
    os.umask(0o077)
    (HOME/'pg-backups').mkdir(mode=0o700, exist_ok=True)
    lock = (HOME/'pg-backups'/'backup.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = (Path(sys.argv[1]).resolve() if len(sys.argv)>1 else
            HOME / 'pg-backups' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if not any(root.is_relative_to(HOME/p) for p in ('pg-backups','ops-backups')):
        raise RuntimeError('Private backup directory required')
    root.mkdir(parents=True, mode=0o700)
    def run(tool, args, filename=None):
        with (root / 'diagnostics.private.log').open('ab') as log:
            result = subprocess.run(client(tool, args, root), stdout=subprocess.PIPE, stderr=log, timeout=1800)
        if result.returncode:
            raise RuntimeError(f'{tool} failed; private diagnostics retained')
        if filename:
            (root / filename).write_bytes(result.stdout)
        return result.stdout
    metadata = run('psql', ['-w', CONNECTION, '-Atc',
        "select json_build_object('server_version',current_setting('server_version'),"
        "'ssl',(select ssl from pg_stat_ssl where pid=pg_backend_pid()),"
        "'schemas',(select json_agg(nspname order by nspname) from pg_namespace where nspname not like 'pg_%' and nspname<>'information_schema'),"
        "'extensions',(select json_agg(extname order by extname) from pg_extension))"], 'source.json')
    # Through a pooler pg_stat_ssl describes its backend link, not the client TLS.
    # All client connections explicitly require certificate+hostname validation.
    run('pg_dump', ['-w', '--dbname='+CONNECTION, '--format=custom',
                    '--lock-wait-timeout=10s', '--file=database.partial.dump'])
    (root / 'database.partial.dump').rename(root / 'database.dump')
    run('pg_dumpall', ['-w', '--database='+CONNECTION, '--roles-only', '--no-role-passwords'], 'roles.sql')
    run('pg_restore', ['--list', 'database.dump'], 'archive-list.private.txt')
    checks = []
    for name in ('database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt'):
        with (root/name).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        checks.append(f'{digest}  {name}\n')
    (root / 'SHA256SUMS').write_text(''.join(checks))
    (root / 'COMPLETE').touch()
    print(json.dumps({'backup': str(root), 'dump_bytes': (root/'database.dump').stat().st_size,
                      'server': json.loads(metadata)['server_version'], 'ssl': True,
                      'client_image': IMAGE}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not expose libpq strings or database content.
        print('PostgreSQL backup incomplete; inspect private diagnostics on VPS')
        raise SystemExit(1)
