"""Read-only backup from the configured v4 container. Binary tar on stdout.

Redirect stdout to a private file; no credentials or record contents are logged.
Run before the first workflow migration while the old API is active.
"""
import hashlib
import io
import json
import sys
import tarfile
from urllib.parse import quote

import requests
from arborscan_v4.corrections_api import _config, _require_private_bucket


def backup():
    _require_private_bucket()
    url, headers, bucket = _config()
    manifest = {}
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        def add(name, raw):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(raw))
            manifest[name] = hashlib.sha256(raw).hexdigest()

        def walk(prefix):
            offset = 0
            while True:
                r = requests.post(url+'/storage/v1/object/list/'+quote(bucket, safe=''),
                    headers=headers, json={'prefix':prefix, 'offset':offset, 'limit':100,
                    'sortBy':{'column':'name','order':'asc'}}, timeout=60)
                r.raise_for_status()
                rows = r.json()
                for row in rows:
                    name = row['name']
                    if '/' in name or name in ('.', '..'): raise ValueError('Unsafe object name')
                    path = prefix+'/'+name
                    if row.get('id') is None:
                        walk(path)
                    else:
                        obj = requests.get(url+'/storage/v1/object/'+quote(bucket, safe='')+'/'+quote(path, safe='/'),
                                           headers=headers, timeout=60)
                        obj.raise_for_status()
                        add('objects/'+path, obj.content)
                if len(rows) < 100: break
                offset += len(rows)
        walk('v4-corrections')
        r = requests.get(url+'/rest/v1/contour_revisions', headers=headers,
                         params={'select':'*', 'limit':'1'}, timeout=30)
        # First rollout only: never claim a partial metadata snapshot is a backup.
        if r.status_code == 404 and r.json().get('code') in ('PGRST205', '42P01'):
            add('metadata-state.json', b'{"contour_revisions":"absent_before_first_migration"}')
        else:
            raise RuntimeError('Metadata may exist; require a database snapshot before proceeding')
        add('manifest.json', json.dumps(manifest, sort_keys=True).encode())


if __name__ == '__main__':
    try:
        backup()
    except Exception:
        sys.stderr.write('Backup failed; discard incomplete archive. No deployment is permitted.\n')
        sys.exit(1)
