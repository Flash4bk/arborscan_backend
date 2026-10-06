"""AS16 live regression: failed revision uploads are not saved contours.

Only the existing owner-scoped index authorizes a schema-2 listing. Old PNGs
remain readable without this index; raw Storage offsets continue pagination.
"""
from copy import deepcopy
from unittest.mock import Mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest
import requests

from arborscan_v4 import corrections_api as api

OWNER='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
ANALYSIS='cccccccc-cccc-4ccc-8ccc-cccccccccccc'


def key(index):
    return ANALYSIS+'_'+format(index,'064x')+'.json'


class Response:
    def __init__(self, rows, status=200):
        self.rows, self.status_code=rows, status
    def json(self):
        return self.rows


@pytest.fixture
def listing(monkeypatch):
    state={'rows': [], 'metadata': [], 'records': {}, 'storage_status': 200}
    config=Mock(return_value=('https://example.invalid',{},'private'))
    storage=Mock(side_effect=lambda *a,**k:Response(state['rows'],state['storage_status']))
    metadata=Mock(side_effect=lambda *a,**k:deepcopy(state['metadata']))
    blobs=Mock(side_effect=lambda owner,identifier:deepcopy(state['records'][identifier]))
    class Store:
        def __init__(self, supplied_config):
            assert supplied_config is config
        request=metadata
    monkeypatch.setattr(api,'_config',config)
    monkeypatch.setattr(api.requests,'post',storage)
    monkeypatch.setattr(api,'WorkflowStore',Store)
    monkeypatch.setattr(api,'_get',blobs)
    state.update(storage=storage,index=metadata,blobs=blobs)
    return state


def test_only_committed_editor_revisions_and_old_pngs_appear(listing):
    legacy,committed,orphan=key(1),key(2),key(3)
    listing['rows']=[{'name':v,'created_at':'synthetic'} for v in (legacy,committed,orphan)]
    listing['metadata']=[{'correction_id':committed}]
    listing['records']={legacy:{'schema_version':1,'editor_state':None},orphan:{'schema_version':2}}
    before=deepcopy(listing['records'])
    result=api.list_corrections(0,OWNER)
    assert [r['correction_id'] for r in result['items']]==[legacy,committed]
    assert result['next_offset'] is None
    # No large original/mask JSON is downloaded for indexed revisions, and no
    # legacy row is registered or automatically accepted merely by listing it.
    assert listing['blobs'].call_count==2
    assert [c.args for c in listing['blobs'].call_args_list]==[(OWNER,legacy),(OWNER,orphan)]
    assert listing['records']==before
    listing['index'].assert_called_once_with('GET','contour_revisions',params={
        'owner_id':'eq.'+OWNER,'select':'correction_id',
        'correction_id':'in.('+','.join((legacy,committed,orphan))+')'})
    assert listing['storage'].call_args.kwargs['json']['prefix']=='v4-corrections/'+OWNER+'/'


def test_all_orphan_page_continues_at_raw_storage_offset(listing):
    listing['rows']=[{'name':key(i)} for i in range(50)]
    listing['records']={key(i):{'schema_version':2} for i in range(50)}
    assert api.list_corrections(20,OWNER)=={'items':[],'next_offset':70}
    assert listing['index'].call_count==1
    assert listing['storage'].call_args.kwargs['json']['offset']==20


def test_full_committed_page_uses_one_index_read_and_no_photo_download(listing):
    listing['rows']=[{'name':key(i)} for i in range(50)]
    listing['metadata']=[{'correction_id':key(i)} for i in range(50)]
    result=api.list_corrections(0,OWNER)
    assert len(result['items'])==50 and result['next_offset']==50
    listing['index'].assert_called_once()
    listing['blobs'].assert_not_called()


def test_old_png_only_history_survives_workflow_database_outage(listing):
    legacy=key(1)
    listing['rows']=[{'name':legacy}]
    listing['records']={legacy:{'schema_version':1}}
    listing['index'].side_effect=HTTPException(503,'Contour workflow unavailable')
    assert api.list_corrections(0,OWNER)=={
        'items':[{'correction_id':legacy,'created_at':None}],'next_offset':None}


def test_outage_cannot_silently_hide_committed_schema2_history(listing):
    legacy,new=key(1),key(2)
    listing['rows']=[{'name':legacy},{'name':new}]
    listing['records']={legacy:{'schema_version':1},new:{'schema_version':2}}
    listing['index'].side_effect=HTTPException(503,'Contour workflow unavailable')
    with pytest.raises(HTTPException) as caught:
        api.list_corrections(0,OWNER)
    assert caught.value.status_code==503


@pytest.mark.parametrize('bad_metadata',[None,{},[{'correction_id':key(99)}]])
def test_invalid_or_unrequested_index_rows_cannot_count_as_commit(listing,bad_metadata):
    listing['rows']=[{'name':key(1)}]
    listing['records']={key(1):{'schema_version':2}}
    listing['metadata']=bad_metadata
    with pytest.raises(HTTPException) as caught:
        api.list_corrections(0,OWNER)
    assert caught.value.status_code==503


def test_storage_failure_is_explicit_and_does_not_fall_back_to_empty_success(listing):
    listing['storage_status']=500
    with pytest.raises(HTTPException) as caught:
        api.list_corrections(0,OWNER)
    assert caught.value.status_code==503
    listing['index'].assert_not_called()
    listing['blobs'].assert_not_called()


def test_unreadable_unindexed_blob_is_explicit_not_silently_discarded(listing):
    listing['rows']=[{'name':key(1)}]
    listing['blobs'].side_effect=HTTPException(503,'Correction storage unavailable')
    with pytest.raises(HTTPException) as caught:
        api.list_corrections(0,OWNER)
    assert caught.value.status_code==503


def test_transport_failure_and_empty_page_preserve_storage_contract(listing):
    assert api.list_corrections(0,OWNER)=={'items':[],'next_offset':None}
    listing['index'].assert_not_called()
    listing['storage'].side_effect=requests.ConnectionError('private connection details')
    with pytest.raises(HTTPException) as caught:
        api.list_corrections(0,OWNER)
    assert caught.value.status_code==503 and 'private' not in caught.value.detail


def test_listing_still_requires_current_server_session(monkeypatch):
    calls=Mock()
    monkeypatch.setattr(api.requests,'post',calls)
    app=FastAPI();app.include_router(api.router)
    response=TestClient(app).get('/v4/corrections')
    assert response.status_code==401
    calls.assert_not_called()
