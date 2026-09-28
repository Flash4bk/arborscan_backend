"""User-local PostgreSQL client wrapper; no server installed on the host."""
import os
from pathlib import Path
import subprocess
import sys
from ops_pg_backup import IMAGE, CONNECTION

tool = sys.argv[1]
if tool not in ('psql', 'pg_dump', 'pg_restore'):
    raise SystemExit('Unsupported PostgreSQL client')
cwd = Path.cwd().resolve()
command = ['docker', 'run', '--rm', '-i', '--user', f'{os.getuid()}:{os.getgid()}',
           '--network', 'none' if tool == 'pg_restore' else 'host',
           '-v', f'{cwd}:{cwd}', '-w', str(cwd)]
if tool != 'pg_restore':
    for name, variable in (('.pg_service.conf', 'PGSERVICEFILE'), ('.pgpass', 'PGPASSFILE')):
        path = Path('/home/arborscan') / name
        command += ['-v', f'{path}:{path}:ro', '-e', f'{variable}={path}']
    command += ['-v', '/home/arborscan/supabase-ca.crt:/home/arborscan/supabase-ca.crt:ro',
                '-e', 'PGCONNECT_TIMEOUT=20']
args = [CONNECTION if a == 'service=arborscan_backup' else
        '--dbname='+CONNECTION if a == '--dbname=service=arborscan_backup' else a
        for a in sys.argv[2:]]
raise SystemExit(subprocess.call(command + [IMAGE, tool] + args))
