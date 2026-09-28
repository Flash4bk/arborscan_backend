"""Install user-local wrappers without replacing other client installations."""
from pathlib import Path
root = Path('/home/arborscan/.local/bin')
root.mkdir(parents=True, exist_ok=True)
for tool in ('psql', 'pg_dump', 'pg_restore'):
    path = root / tool
    content = f'#!/bin/sh\n# ArborScan pinned PostgreSQL client\nexec python3 /home/arborscan/ops-tools/ops_pg_client.py {tool} "$@"\n'
    if path.exists() and 'ArborScan pinned PostgreSQL client' not in path.read_text():
        raise SystemExit('Existing client wrapper preserved; installation stopped')
    path.write_text(content)
    path.chmod(0o700)
print('User-local psql, pg_dump, pg_restore wrappers installed')
