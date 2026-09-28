"""Print classifications only; never connection strings or raw server errors."""
from pathlib import Path
import json
path = max(Path('/home/arborscan/pg-backups').rglob('diagnostics.private.log'), key=lambda p:p.stat().st_mtime)
root = path.parent
text = path.read_text().lower()
print(json.dumps({'backup': root.name, 'files': [p.name for p in root.iterdir()],
                  'error_categories': [key for key in (
                      'certificate', 'password authentication failed', 'permission denied',
                      'root certificate', 'invalid', 'no password supplied', 'does not exist',
                      'could not', 'unrecognized', 'option', 'already exists', 'extension',
                      'vault', 'role', 'public', 'permission', 'transaction_timeout', 'syntax') if key in text]}))
