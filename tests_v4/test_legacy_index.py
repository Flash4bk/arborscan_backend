from arborscan_v4 import index_legacy_contours as indexer


def test_legacy_index_is_explicit_and_does_not_reset_decisions(monkeypatch):
    api=indexer.api
    owner='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    rows={}
    class Store:
        def __init__(self,*args): pass
        def ready(self): pass
        def get(self,o,k): return rows.get((o,k))
        def transition(self,action,o,k,**fields):
            assert action=='legacy'
            rows[o,k]={'status':'submitted'}
            return rows[o,k]
    class Response:
        status_code=200
        def json(self): return [{'name':owner}]
    monkeypatch.setattr(api,'WorkflowStore',Store)
    monkeypatch.setattr(api,'_require_private_bucket',lambda:None)
    monkeypatch.setattr(api,'_config',lambda:('https://example.invalid',{},'private'))
    monkeypatch.setattr(indexer.requests,'post',lambda *a,**k:Response())
    monkeypatch.setattr(api,'list_corrections',lambda **kwargs:{'items':[{'correction_id':'legacy'}],'next_offset':None})
    monkeypatch.setattr(api,'_get',lambda *args:{'schema_version':1,'analysis_id':'analysis','image_sha256':'hash'})
    assert indexer.index_legacy()['legacy_candidates']==1
    assert rows=={}
    assert indexer.index_legacy(True)['indexed']==1
    rows[owner,'legacy']['status']='accepted'
    assert indexer.index_legacy(True)['indexed']==0
    assert rows[owner,'legacy']['status']=='accepted'
