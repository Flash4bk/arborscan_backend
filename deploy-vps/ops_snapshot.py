"""Private read-only application data snapshot, NOT a transactional pg_dump.

Run inside the configured container, redirect stdout to a mode-600 .partial tar.
All bucket objects and exposed public tables are copied. No credentials in output
except private table contents inside the archive. Diagnostics go only to stderr.
"""
import hashlib
import io
import json
import sys
import tarfile
import time
from urllib.parse import quote
import requests
from arborscan_v4.corrections_api import _config


def snapshot(output):
    url, headers, _ = _config()
    session = requests.Session()
    session.headers.update(headers)
    started = time.time()
    hashes = {}
    counts = {}
    def fetch(path, method='GET', **kwargs):
        r = session.request(method, url+path, timeout=(15, 120), **kwargs)
        r.raise_for_status()
        return r
    def rows(table):
        result = []
        offset = 0
        while True:
            page = fetch('/rest/v1/'+quote(table, safe=''), params={
                'select': '*', 'offset': offset, 'limit': 500}).json()
            if not isinstance(page, list):
                raise ValueError('Invalid table response')
            result.extend(page)
            if not page: break
            offset += len(page)
        return sorted(result, key=lambda row: json.dumps(row, sort_keys=True))
    with tarfile.open(fileobj=output, mode='w|') as archive:
        def add(name, raw):
            item = tarfile.TarInfo(name)
            item.mode = 0o600
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
            hashes[name] = hashlib.sha256(raw).hexdigest()
        def add_json(name, value):
            add(name, json.dumps(value, sort_keys=True).encode())
        spec = fetch('/rest/v1/', headers={**headers, 'Accept': 'application/openapi+json'}).json()
        tables = sorted(p[1:] for p, verbs in spec['paths'].items()
                        if p.startswith('/') and not p.startswith('/rpc/') and 'get' in verbs and p != '/')
        snapshots = {}
        add_json('schema/postgrest-openapi.json', spec)
        versions = {}
        for rpc in ('contour_workflow_version', 'server_history_version', 'model_quality_version'):
            versions[rpc] = fetch('/rest/v1/rpc/'+rpc, 'POST', json={}).json()
        buckets = fetch('/storage/v1/bucket').json()
        add_json('schema/buckets.json', buckets)
        object_count = 0
        def walk(bucket, prefix=''):
            nonlocal object_count
            offset = 0
            while True:
                page = fetch('/storage/v1/object/list/'+quote(bucket, safe=''), 'POST', json={
                    'prefix': prefix, 'limit': 100, 'offset': offset,
                    'sortBy': {'column': 'name', 'order': 'asc'}}).json()
                for row in page:
                    name = row['name']
                    if '/' in name or '\\' in name or name in ('.', '..'): raise ValueError('Unsafe object')
                    key = prefix+'/'+name if prefix else name
                    if row.get('id') is None:
                        walk(bucket, key)
                    else:
                        # Stream to bounded temporary disk rather than retaining all objects.
                        import tempfile
                        with session.get(url+'/storage/v1/object/'+quote(bucket, safe='')+'/'+quote(key, safe='/'),
                                         timeout=(15, 120), stream=True) as r:
                            r.raise_for_status()
                            with tempfile.TemporaryFile() as tmp:
                                digest = hashlib.sha256(); size = 0
                                for chunk in r.iter_content(1024*1024):
                                    tmp.write(chunk); digest.update(chunk); size += len(chunk)
                                tmp.seek(0)
                                target = 'objects/'+bucket+'/'+key
                                info = tarfile.TarInfo(target); info.mode = 0o600; info.size = size
                                archive.addfile(info, tmp)
                                hashes[target] = digest.hexdigest(); object_count += 1
                if not page: break
                offset += len(page)
        for bucket in buckets:
            name = bucket['id']
            if '/' in name or '..' in name: raise ValueError('Unsafe bucket')
            walk(name)
        for table in tables:
            if '/' in table or '..' in table: raise ValueError('Unsafe table')
            data = rows(table)
            snapshots[table] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            counts[table] = len(data)
            add_json('tables/'+table+'.json', data)
        # Detect concurrent table changes; no claim of a transactionally consistent DB backup.
        for table, before in snapshots.items():
            after = hashlib.sha256(json.dumps(rows(table), sort_keys=True).encode()).hexdigest()
            if before != after: raise RuntimeError('Tables changed during backup; retry later')
        add_json('snapshot.json', {'format': 1, 'started_at_unix': started,
            'finished_at_unix': time.time(), 'tables': counts, 'objects': object_count,
            'schema_versions': versions, 'transactional_database_backup': False,
            'excludes': ['Postgres roles/functions/triggers/RLS not in OpenAPI', 'Supabase internal schemas',
                         'VPS files/models/configuration (separate archive)']})
        add_json('manifest.json', hashes.copy())
    session.close()


if __name__ == '__main__':
    try:
        snapshot(sys.stdout.buffer)
    except Exception:
        print('Snapshot failed; retain .partial for diagnosis, do not publish as complete.', file=sys.stderr)
        sys.exit(1)
