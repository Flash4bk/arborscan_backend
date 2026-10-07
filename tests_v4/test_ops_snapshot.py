"""Read-only backup coverage, bounded retries and private failure diagnostics."""
from contextlib import contextmanager
import json
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_snapshot as ops
import ops_verify_restore

SECRET = 'private-owner-and-token-do-not-print'


class Page:
    def __init__(self, rows, total, offset=0):
        self.rows = rows
        self.headers = {'Content-Range': ('%d-%d' % (offset, offset+len(rows)-1) if rows else '*') + '/' + str(total)}

    def json(self):
        return self.rows


def failure(reason='statement_timeout', status=500):
    diagnostic = ops.ReadDiagnostics();diagnostic.reset('table_snapshot')
    diagnostic.statuses.append(status);diagnostic.reason = reason
    return ops.SnapshotReadError(diagnostic)


@contextmanager
def http_fixture(responses):
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests_seen.append(self.path)
            status, body = responses[min(len(requests_seen)-1, len(responses)-1)]
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers();self.wfile.write(raw)

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    session, diagnostic = ops.reader_session({'Authorization':'Bearer '+SECRET})
    session.mount('http://', session.get_adapter('https://example.test'))
    try:
        yield session, diagnostic, 'http://127.0.0.1:'+str(server.server_port), requests_seen
    finally:
        session.close();server.shutdown();server.server_close();thread.join(timeout=2)


class SnapshotReadTest(unittest.TestCase):
    def full_snapshot(self, changed=False):
        reads = []
        class Response:
            status_code=200
            def __init__(self, body):
                self.body=body
                self.headers={'Content-Range':'0-0/1'}
            def json(self):return self.body
            def raise_for_status(self):pass
            def iter_content(self, *args):yield self.body
            def __enter__(self):return self
            def __exit__(self, *args):pass
        class Session:
            def request(self, method, url, **kwargs):
                path=url.split('https://test.invalid',1)[1]
                if path=='/rest/v1/':
                    return Response({'paths':{'/analyses':{'get':{}}},'definitions':{'analyses':{'properties':{'id':{'description':'Primary Key <pk/>'}}}}})
                if path.startswith('/rest/v1/rpc/'):return Response(1)
                if path=='/storage/v1/bucket':return Response([{'id':'demo-bucket'}])
                if path=='/storage/v1/object/list/demo-bucket':return Response([{'name':'control.bin','id':'synthetic'}])
                if path=='/storage/v1/object/demo-bucket/control.bin':return Response(b'synthetic private original bytes')
                if path=='/rest/v1/analyses':
                    reads.append(kwargs['params'])
                    return Response([{'id':1,'value':'second' if changed and len(reads)==2 else 'original'}])
                raise AssertionError('Unexpected synthetic request')
            def close(self):pass
        d=ops.ReadDiagnostics();d.reset('unknown')
        fake_config=SimpleNamespace(_config=lambda:('https://test.invalid',{},'demo-bucket'))
        output=io.BytesIO()
        with patch.dict(sys.modules,{'arborscan_v4.corrections_api':fake_config}),patch.object(ops,'reader_session',return_value=(Session(),d)):
            if changed:
                with self.assertRaisesRegex(RuntimeError,'Tables changed'):ops.snapshot(output)
            else:ops.snapshot(output)
        return output.getvalue(),reads

    def test_complete_new_reader_snapshot_keeps_existing_offline_restore_format(self):
        raw,reads=self.full_snapshot()
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'application.tar';source.write_bytes(raw)
            result=ops_verify_restore.restore(source,Path(temp)/'restored')
            self.assertEqual(result['files_restored'],5)
            self.assertFalse(result['database_restored'])
            self.assertEqual((Path(temp)/'restored/objects/demo-bucket/control.bin').read_bytes(),b'synthetic private original bytes')
        self.assertEqual(len(reads),2)
        self.assertTrue(all(p['limit']==100 and p['order']=='id.asc' for p in reads))

    def test_changed_table_cannot_publish_a_complete_snapshot_manifest(self):
        raw,reads=self.full_snapshot(changed=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            self.assertNotIn('manifest.json',archive.getnames())
        self.assertEqual(len(reads),2)

    def test_smaller_stable_pages_cover_every_row_once(self):
        rows = [{'id':i,'photo':SECRET+str(i)} for i in range(67)]
        calls = []
        def fetch(path, **kwargs):
            p=kwargs['params'];calls.append((p['offset'],p['limit']))
            self.assertEqual(p['order'], 'id.asc')
            self.assertEqual(kwargs['headers'], {'Prefer':'count=exact'})
            if p['limit']==100:raise failure()
            return Page(rows[p['offset']:p['offset']+p['limit']],len(rows),p['offset'])
        result=ops.table_rows(fetch,'analyses',['id'])
        self.assertEqual({r['id'] for r in result}, set(range(67)))
        self.assertEqual(len(result),67)
        self.assertEqual(calls,[(0,100),(0,25),(25,25),(50,25)])

    def test_timeout_after_first_page_keeps_the_same_offset(self):
        rows=[{'id':i} for i in range(131)];calls=[]
        def fetch(path, **kwargs):
            p=kwargs['params'];calls.append((p['offset'],p['limit']))
            if p['offset']==100 and p['limit']==100:raise failure()
            return Page(rows[p['offset']:p['offset']+p['limit']],len(rows),p['offset'])
        self.assertEqual(len(ops.table_rows(fetch,'analyses',['id'])),131)
        self.assertEqual(calls,[(0,100),(100,100),(100,25),(125,25)])

    def test_persistent_timeout_fails_at_one_row_without_skipping(self):
        calls=[]
        def fetch(path, **kwargs):
            calls.append(kwargs['params']['limit']);raise failure()
        with self.assertRaises(ops.SnapshotReadError):ops.table_rows(fetch,'analyses',['id'])
        self.assertEqual(calls,[100,25,5,1])

    def test_other_http_error_does_not_shrink_or_skip_the_table(self):
        calls=[]
        def fetch(path, **kwargs):
            calls.append(kwargs);raise failure('permission_denied',403)
        with self.assertRaises(ops.SnapshotReadError):ops.table_rows(fetch,'analyses',['id'])
        self.assertEqual(len(calls),1)

    def test_composite_primary_key_is_ordered_and_deduplicated(self):
        rows=[{'owner_id':1,'version_id':i} for i in range(104)]
        def fetch(path, **kwargs):
            p=kwargs['params'];self.assertEqual(p['order'],'owner_id.asc,version_id.asc')
            page=rows[:100] if p['offset']==0 else [rows[99],*rows[100:]]
            return Page(page,len(rows),p['offset'])
        with self.assertRaisesRegex(ValueError,'Repeated primary key'):
            ops.table_rows(fetch,'report_versions',['owner_id','version_id'])

    def test_changed_exact_count_refuses_the_snapshot(self):
        rows=[{'id':i} for i in range(104)]
        def fetch(path, **kwargs):
            p=kwargs['params'];return Page(rows[p['offset']:p['offset']+p['limit']],104 if p['offset']==0 else 105,p['offset'])
        with self.assertRaisesRegex(ValueError,'changed'):
            ops.table_rows(fetch,'analyses',['id'])

    def test_server_page_cap_cannot_silently_drop_rows(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            ops.table_rows(lambda *args,**kwargs:Page([{'id':i} for i in range(25)],99),'analyses',['id'])

    def test_missing_exact_count_is_not_claimed_as_a_complete_table(self):
        p=Page([],0);p.headers={}
        with self.assertRaisesRegex(ValueError,'count'):
            ops.table_rows(lambda *args,**kwargs:p,'analyses',['id'])

    def test_missing_or_unsafe_primary_key_refuses_before_read(self):
        for keys in ([],['id;secret'],['id.desc']):
            with self.subTest(keys=keys),self.assertRaisesRegex(ValueError,'primary key'):
                ops.table_rows(lambda *args,**kwargs:self.fail('No read allowed'),'analyses',keys)

    def test_empty_table_is_valid_only_with_exact_zero_count(self):
        self.assertEqual(ops.table_rows(lambda *args,**kwargs:Page([],0),'analyses',['id']),[])

    def test_storage_transient_500_is_still_retried_and_can_succeed(self):
        with http_fixture([(500,{'message':'temporary '+SECRET}),(200,{'ok':True})]) as (session,d,u,calls),patch.object(ops.time,'sleep'):
            r=ops.read(session,d,u,'/objects/'+SECRET,stage='storage_object')
            self.assertEqual(r.json(),{'ok':True});self.assertEqual(len(calls),2)
            self.assertEqual(d.statuses,[500,200])

    def test_exhausted_retry_keeps_status_and_stage_without_private_details(self):
        with http_fixture([(500,{'message':'server error '+SECRET})]) as (session,d,u,calls),patch.object(ops.time,'sleep'):
            with self.assertRaises(ops.SnapshotReadError) as caught:
                ops.read(session,d,u,'/objects/'+SECRET,stage='storage_object')
            report=caught.exception.summary();self.assertEqual(len(calls),4)
            self.assertEqual(report['http_status'],500);self.assertEqual(report['status_history'],[500]*4)
            self.assertEqual(report['stage'],'storage_object')
            self.assertNotIn(SECRET,json.dumps(report));self.assertNotIn('http',str(caught.exception).lower())

    def test_statement_timeout_is_observed_once_before_page_adaptation(self):
        with http_fixture([(500,{'message':'canceling statement due to statement timeout; '+SECRET})]) as (session,d,u,calls):
            with self.assertRaises(ops.SnapshotReadError) as caught:
                ops.read(session,d,u,'/analyses?token='+SECRET,stage='table_snapshot')
            self.assertEqual(len(calls),1);self.assertEqual(caught.exception.reason,'statement_timeout')
            self.assertEqual(caught.exception.status,500);self.assertNotIn(SECRET,json.dumps(caught.exception.summary()))

    def test_429_connection_pressure_is_reported_separately_from_sql_timeout(self):
        with http_fixture([(429,{'message':'Too many connections '+SECRET})]) as (session,d,u,calls),patch.object(ops.time,'sleep'):
            with self.assertRaises(ops.SnapshotReadError) as caught:
                ops.read(session,d,u,'/list/'+SECRET,'POST','storage_list')
            self.assertEqual(len(calls),4);self.assertEqual(caught.exception.reason,'too_many_connections')
            self.assertEqual(caught.exception.status,429);self.assertEqual(caught.exception.stage,'storage_list')

    def test_authorization_error_is_not_retried(self):
        with http_fixture([(401,{'message':SECRET})]) as (session,d,u,calls):
            with self.assertRaises(ops.SnapshotReadError) as caught:
                ops.read(session,d,u,'/list/'+SECRET,stage='storage_list')
            self.assertEqual(len(calls),1);self.assertEqual(caught.exception.status,401)
            self.assertNotIn(SECRET,json.dumps(caught.exception.summary()))


if __name__=='__main__':unittest.main()
