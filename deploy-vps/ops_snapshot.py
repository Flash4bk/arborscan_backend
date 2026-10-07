"""Private read-only application data snapshot, NOT a transactional pg_dump.

Run inside the configured container, redirect stdout to a mode-600 .partial tar.
All bucket objects and exposed public tables are copied. No credentials in output
except private table contents inside the archive. Diagnostics go only to stderr.
"""
import hashlib
import io
import json
import re
import sys
import tarfile
import time
from urllib.parse import quote
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STAGES = {'schema_inventory', 'schema_versions', 'bucket_inventory',
          'storage_list', 'storage_object', 'table_snapshot'}
TABLE_PAGE_SIZES = (100, 25, 5, 1)


def safe_reason(body):
    body = body.lower()
    for phrase, reason in (('statement timeout', 'statement_timeout'),
                           ('too many connections', 'too_many_connections'),
                           ('permission denied', 'permission_denied')):
        if phrase in body:
            return reason
    return 'transient_http'


class ReadDiagnostics:
    def reset(self, stage):
        self.stage = stage if stage in STAGES else 'unknown'
        self.statuses = []
        self.reason = None


class SnapshotReadError(Exception):
    """Only allow-listed operational metadata; never a URL or server body."""
    def __init__(self, diagnostic):
        self.stage = diagnostic.stage
        self.statuses = diagnostic.statuses[:4]
        self.status = self.statuses[-1] if self.statuses else None
        self.reason = diagnostic.reason or 'network_error'
        super().__init__('Private snapshot read failed')

    def summary(self):
        return {'stage': self.stage, 'http_status': self.status,
                'status_history': self.statuses, 'attempts': len(self.statuses),
                'reason': self.reason}


class SnapshotRetry(Retry):
    def __init__(self, *args, diagnostic=None, **kwargs):
        self.diagnostic = diagnostic
        super().__init__(*args, **kwargs)

    def new(self, **kwargs):
        kwargs['diagnostic'] = self.diagnostic
        return super().new(**kwargs)

    def increment(self, method=None, url=None, response=None, error=None, *args, **kwargs):
        diagnostic = self.diagnostic
        diagnostic.statuses.append(response.status if response is not None else None)
        if response is not None and response.status >= 400:
            # Retry discards this failed response. Inspect only a bounded prefix
            # to classify known errors; never log it or the URL/headers.
            body = response.read(4096, decode_content=True).decode('utf-8', errors='replace')
            diagnostic.reason = safe_reason(body)
            if diagnostic.stage == 'table_snapshot' and diagnostic.reason == 'statement_timeout':
                response.drain_conn(); response.release_conn()
                # An identical heavy SQL page timed out. Shrink its size, not
                # the retry count or production SQL timeout/index settings.
                raise SnapshotReadError(diagnostic)
        return super().increment(method, url, response, error, *args, **kwargs)


def reader_session(headers):
    diagnostic = ReadDiagnostics()
    diagnostic.reset('unknown')
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=SnapshotRetry(
        diagnostic=diagnostic, total=3, connect=3, read=3, status=3, backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET', 'POST'),
        respect_retry_after_header=False)))
    session.headers.update(headers)
    return session, diagnostic


def read(session, diagnostic, url, path, method='GET', stage='unknown', **kwargs):
    diagnostic.reset(stage)
    try:
        response = session.request(method, url+path, timeout=(15, 120), **kwargs)
        diagnostic.statuses.append(response.status_code)
        response.raise_for_status()
        return response
    except SnapshotReadError:
        raise
    except requests.RequestException as error:
        if getattr(error, 'response', None) is not None:
            diagnostic.reason = safe_reason(error.response.text[:4096])
        raise SnapshotReadError(diagnostic) from None


def table_rows(fetch, table, keys):
    """Bounded pages, stable primary-key order, exact coverage and no repeats."""
    if not keys or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) for key in keys):
        raise ValueError('A stable primary key is required for the snapshot')
    result = []
    seen = set()
    offset = 0
    size_index = 0
    expected_total = None
    while True:
        size = TABLE_PAGE_SIZES[size_index]
        try:
            response = fetch('/rest/v1/'+quote(table, safe=''), stage='table_snapshot',
                headers={'Prefer': 'count=exact'}, params={'select': '*', 'offset': offset,
                    'limit': size, 'order': ','.join(key+'.asc' for key in keys)})
        except SnapshotReadError as error:
            if error.reason != 'statement_timeout' or size_index == len(TABLE_PAGE_SIZES)-1:
                raise
            size_index += 1
            continue  # Keep exactly the same offset until a page succeeds.
        page = response.json()
        if not isinstance(page, list):
            raise ValueError('Invalid table response')
        content_range = response.headers.get('Content-Range', '')
        total = re.fullmatch(r'(?:\d+-\d+|\*)/(\d+)', content_range)
        if total is None:
            raise ValueError('Exact table count was not confirmed')
        actual_total = int(total[1])
        if expected_total is None:
            expected_total = actual_total
        elif expected_total != actual_total:
            raise ValueError('Table changed during backup')
        for row in page:
            identity = tuple(row[key] for key in keys)
            if identity in seen:
                raise ValueError('Repeated primary key in snapshot')
            seen.add(identity)
        result.extend(page)
        offset += len(page)
        if offset == expected_total:
            break
        if len(page) < size:
            if offset != expected_total:
                raise ValueError('Incomplete table snapshot')
            break
    return sorted(result, key=lambda row: json.dumps(row, sort_keys=True))


def snapshot(output):
    from arborscan_v4.corrections_api import _config
    url, headers, _ = _config()
    # This session only reads: POST is used by list/version RPC endpoints.
    # Keep the existing bounded transient Storage retry policy.
    session, diagnostic = reader_session(headers)
    started = time.time()
    hashes = {}
    counts = {}
    def fetch(path, method='GET', stage='unknown', **kwargs):
        return read(session, diagnostic, url, path, method, stage, **kwargs)
    def rows(table):
        keys = [key for key, value in spec.get('definitions', {}).get(table, {}).get('properties', {}).items()
                if '<pk/>' in value.get('description', '')]
        return table_rows(fetch, table, keys)
    with tarfile.open(fileobj=output, mode='w|') as archive:
        def add(name, raw):
            item = tarfile.TarInfo(name)
            item.mode = 0o600
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
            hashes[name] = hashlib.sha256(raw).hexdigest()
        def add_json(name, value):
            add(name, json.dumps(value, sort_keys=True).encode())
        spec = fetch('/rest/v1/', stage='schema_inventory', headers={**headers, 'Accept': 'application/openapi+json'}).json()
        tables = sorted(p[1:] for p, verbs in spec['paths'].items()
                        if p.startswith('/') and not p.startswith('/rpc/') and 'get' in verbs and p != '/')
        snapshots = {}
        add_json('schema/postgrest-openapi.json', spec)
        versions = {}
        for rpc in ('contour_workflow_version', 'server_history_version', 'model_quality_version'):
            versions[rpc] = fetch('/rest/v1/rpc/'+rpc, 'POST', stage='schema_versions', json={}).json()
        buckets = fetch('/storage/v1/bucket', stage='bucket_inventory').json()
        add_json('schema/buckets.json', buckets)
        object_count = 0
        def walk(bucket, prefix=''):
            nonlocal object_count
            offset = 0
            while True:
                page = fetch('/storage/v1/object/list/'+quote(bucket, safe=''), 'POST', stage='storage_list', json={
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
                        with fetch('/storage/v1/object/'+quote(bucket, safe='')+'/'+quote(key, safe='/'),
                                   stage='storage_object', stream=True) as r:
                            with tempfile.TemporaryFile() as tmp:
                                digest = hashlib.sha256(); size = 0
                                for chunk in r.iter_content(1024*1024):
                                    tmp.write(chunk); digest.update(chunk); size += len(chunk)
                                tmp.seek(0)
                                target = 'objects/'+bucket+'/'+key
                                info = tarfile.TarInfo(target); info.mode = 0o600; info.size = size
                                archive.addfile(info, tmp)
                                hashes[target] = digest.hexdigest(); object_count += 1
                if len(page) < 100: break
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
    except Exception as error:
        # Never log exception text: requests errors can contain private object URLs.
        metadata = error.summary() if isinstance(error, SnapshotReadError) else {}
        print(json.dumps({'event': 'snapshot_failed', 'error_type':type(error).__name__, **metadata}), file=sys.stderr)
        print(f'Snapshot failed ({type(error).__name__}, HTTP {metadata.get("http_status")}); '
              'retain .partial for diagnosis, do not publish as complete.', file=sys.stderr)
        sys.exit(1)
