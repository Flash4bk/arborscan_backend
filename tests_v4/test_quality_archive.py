import json
from copy import deepcopy
import pytest
from fastapi import HTTPException
from arborscan_v4 import model_quality_api as q


def test_json_only_bucket_chunk_roundtrip_retry_and_corruption(monkeypatch):
    stored={}
    class Response:
        def __init__(self,status,payload=None):self.status_code=status;self.payload=payload
        def json(self):return deepcopy(self.payload)
    def post(url,headers,data,timeout):
        assert headers['Content-Type']=='application/json'
        assert headers['x-upsert']=='false'
        if url in stored:return Response(409)
        stored[url]=json.loads(data);return Response(200)
    def get(url,**kwargs):return Response(200,stored[url]) if url in stored else Response(404)
    monkeypatch.setattr(q,'_config',lambda:('https://test.invalid',{},'private'))
    monkeypatch.setattr(q,'ARCHIVE_CHUNK_BYTES',64)
    monkeypatch.setattr(q,'MAX_ARCHIVE_BYTES',256)
    monkeypatch.setattr(q.requests,'post',post);monkeypatch.setattr(q.requests,'get',get)
    raw=bytes(range(200));digest=q.sha(raw)
    assert q.asset(digest,raw)==raw
    assert len(stored)==5 # Four small JSON parts and a manifest.
    assert q.asset(digest)==raw
    assert q.asset(digest,raw)==raw and len(stored)==5
    first=next(k for k in stored if k.endswith('part-000.json'))
    stored[first]['base64']='AA=='
    with pytest.raises(HTTPException):q.asset(digest)
    with pytest.raises(HTTPException):q.asset(digest,raw) # Never overwrite a corrupt existing object.


def test_failed_part_never_publishes_index(monkeypatch):
    calls=[]
    monkeypatch.setattr(q,'_config',lambda:('https://test.invalid',{},'private'))
    def post(url,**kwargs):
        calls.append(url)
        return type('Response',(),{'status_code':500})()
    monkeypatch.setattr(q.requests,'post',post)
    with pytest.raises(HTTPException):q.asset(q.sha(b'fixture'),b'fixture')
    assert len(calls)==1 and calls[0].endswith('part-000.json')
