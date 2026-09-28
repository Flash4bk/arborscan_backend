"""Read-only credential discovery. Prints paths/key names, never values/DSNs.

Run on the VPS via stdin. No config modifications or connection attempts.
"""
import json
import base64
import os
from pathlib import Path
import re
import subprocess

pattern = re.compile(r'postgres(?:ql)?://|\b(?:PGPASSWORD|DATABASE_URL|DB_PASSWORD|POSTGRES_PASSWORD|PGSERVICE|POSTGRES_URL)\b', re.I)
roots = [Path('/opt/arborscan'), Path('/etc/arborscan')]
skip = {'.git', 'node_modules', 'env', '.venv', 'venv', 'build', '.dart_tool',
        '__pycache__', 'models', 'output', 'runs', 'raw_data', 'dataset_yolov8'}
matches = []
for root in roots:
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip and not Path(base, d).is_symlink()]
        for name in files:
            path = Path(base, name)
            if path.is_symlink() or path.stat().st_size > 2_000_000:
                continue
            if not ('.env' in name or path.suffix in ('.conf', '.ini', '.toml', '.yaml', '.yml', '.py', '.sh')):
                continue
            if pattern.search(path.read_text(errors='replace')):
                matches.append(str(path))
print(json.dumps({'candidate_config_files': matches}))
for name in ('arborscan-api', 'arborscan-api-v4', 'arborscan-quality-worker'):
    raw = subprocess.check_output(['docker', 'inspect', name], stderr=subprocess.DEVNULL)
    env = json.loads(raw)[0]['Config'].get('Env', [])
    keys = [entry.split('=', 1)[0] for entry in env if pattern.search(entry)]
    print(json.dumps({'container': name, 'candidate_environment_keys': keys}))
    values = dict(entry.split('=', 1) for entry in env if '=' in entry)
    key = values.get('SUPABASE_SERVICE_KEY') or values.get('SUPABASE_SERVICE_ROLE_KEY') or ''
    try:
        claim = json.loads(base64.urlsafe_b64decode(key.split('.')[1] + '==='))
        role = claim.get('role')
        role = role if role in ('anon', 'authenticated', 'service_role') else 'unknown'
    except (ValueError, IndexError, UnicodeError):
        role = 'opaque_or_unavailable'
    print(json.dumps({'container': name, 'configured_api_key_role': role}))
for name in ('.pg_service.conf', '.pgpass'):
    path = Path('/home/arborscan', name)
    print(json.dumps({'file': str(path), 'exists': path.exists(),
                      'mode': oct(path.stat().st_mode & 0o777) if path.exists() else None}))
