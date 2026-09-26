import asyncio
import importlib.util
import io
import json
import hashlib
import tarfile
from pathlib import Path
from unittest.mock import Mock
import pytest
from arborscan_v4.reliability import RequestLimits


def test_capacity_release_after_disconnect_and_no_extra_work():
    async def scenario():
        entered = asyncio.Event(); release = asyncio.Event(); messages = []
        async def app(scope, receive, send):
            entered.set(); await release.wait(); raise RuntimeError('disconnect')
        limiter = RequestLimits(app, heavy_limit=1)
        scope = {'type':'http','method':'POST','headers':[]}
        async def receive(): return {'type':'http.request','body':b''}
        async def send(message): messages.append(message)
        first = asyncio.create_task(limiter(scope, receive, send))
        await entered.wait()
        await limiter(scope, receive, send)
        assert messages[0]['status'] == 429
        release.set()
        with pytest.raises(RuntimeError): await first
        assert limiter.active == 0
    asyncio.run(scenario())


def test_oversized_request_rejected_before_app_and_chunked_limit():
    async def scenario():
        calls = []
        async def app(scope, receive, send):
            calls.append(1); await receive()
        limiter = RequestLimits(app, max_bytes=3)
        messages=[]
        async def send(m): messages.append(m)
        async def receive(): return {'type':'http.request','body':b'1234'}
        await limiter({'type':'http','method':'POST','headers':[(b'content-length',b'4')]},receive,send)
        assert not calls and messages[0]['status']==413
        assert dict(messages[0]['headers'])[b'x-request-id']
        from starlette.exceptions import HTTPException
        with pytest.raises(HTTPException) as e:
            await limiter({'type':'http','method':'POST','headers':[]},receive,send)
        assert e.value.status_code==413 and limiter.active==0
    asyncio.run(scenario())


def test_dependency_failure_is_not_readiness(monkeypatch):
    from arborscan_v4 import corrections_api as c, correction_workflow as w
    from arborscan_v4.reliability import dependency_status
    monkeypatch.setattr(c,'_config',lambda:('unreachable',{},'private'))
    monkeypatch.setattr(w.WorkflowStore,'request',Mock(side_effect=RuntimeError('secret must not surface')))
    monkeypatch.setattr(c,'_require_private_bucket',Mock(side_effect=RuntimeError('secret')))
    assert not any(dependency_status().values())


def test_offline_restore_checks_bytes_and_rejects_corruption(tmp_path):
    spec=importlib.util.spec_from_file_location('ops_restore',Path(__file__).parents[1]/'deploy-vps/ops_verify_restore.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def archive(path, raw, digest):
        with tarfile.open(path,'w') as tar:
            for name,data in [('objects/test/file',raw),('manifest.json',json.dumps({'objects/test/file':digest}).encode())]:
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    source=tmp_path/'copy.tar';raw=b'actual restored file'
    archive(source,raw,hashlib.sha256(raw).hexdigest())
    result=module.restore(source,tmp_path/'restored')
    assert (tmp_path/'restored/objects/test/file').read_bytes()==raw
    assert result['files_restored']==1 and not result['database_restored']
    archive(source,b'damaged',hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError,match='Checksum'):module.restore(source,tmp_path/'bad')
