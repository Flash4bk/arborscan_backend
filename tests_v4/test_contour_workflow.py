import base64
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'arborscan_v4'))
import corrections_api as api
from correction_workflow import validate_editor, WorkflowStore

OWNER='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
OTHER='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
AID='cccccccc-cccc-4ccc-8ccc-cccccccccccc'

@pytest.fixture
def storage(monkeypatch):
    records, metadata = {}, {}
    class Store:
        def __init__(self,*args): pass
        def ready(self): pass
        def get(self,owner,key): return metadata.get((owner,key))
        def transition(self,action,owner,key,**fields):
            if (owner,key) in metadata: return metadata[owner,key]
            if fields.get('parent') and any(o==owner and m.get('parent_id')==fields['parent'] for (o,k),m in metadata.items()):
                raise HTTPException(409,'Conflict')
            row={'status':'submitted' if action=='legacy' else 'draft','decisions':[], 'parent_id':fields.get('parent')}
            metadata[owner,key]=row
            return row
    monkeypatch.setattr(api,'WorkflowStore',Store)
    monkeypatch.setattr(api,'_put',lambda owner,key,record: records.setdefault((owner,key),copy.deepcopy(record)))
    def get(owner,key):
        if (owner,key) not in records: raise HTTPException(404,'Not found')
        return copy.deepcopy(records[owner,key])
    monkeypatch.setattr(api,'_get',get)
    return records,metadata

@pytest.fixture
def payload():
    image=cv2.imencode('.jpg',np.full((64,48,3),150,np.uint8))[1].tobytes()
    mask=np.zeros((64,48),np.uint8); mask[5:55,10:30]=255
    png=cv2.imencode('.png',mask)[1].tobytes()
    state={'version':1,'coordinates':'normalized_oriented_image','width':48,'height':64,
           'closed':True,'points':[{'x':.2,'y':.1},{'x':.8,'y':.1},{'x':.5,'y':.8}]}
    return image,png,state

def test_state_roundtrip_retry_and_parent(storage,payload):
    image,mask,state=payload
    first=api._validate_and_save(OWNER,AID,image,mask,json.dumps(state))
    assert first==api._validate_and_save(OWNER,AID,image,mask,json.dumps(state,indent=2))
    record=api.get_correction(first['correction_id'],OWNER)
    assert record['editor_state']==state
    assert base64.b64decode(record['original_image_base64'])==image
    assert record['eligible_for_training'] is False
    storage[1][OWNER,first['correction_id']]['status']='accepted'
    state['points'][0]['x']=.3
    second=api._validate_and_save(OWNER,AID,image,mask,json.dumps(state),first['correction_id'])
    assert second['review_status']=='draft'
    assert second['parent_id']==first['correction_id']
    state['points'][0]['x']=.4
    with pytest.raises(HTTPException) as conflict:
        api._validate_and_save(OWNER,AID,image,mask,json.dumps(state),first['correction_id'])
    assert conflict.value.status_code==409

def test_legacy_and_owner_isolation(storage,payload):
    image,mask,state=payload
    old=api._validate_and_save(OWNER,AID,image,mask)
    assert api.get_correction(old['correction_id'],OWNER).get('editor_state') is None
    with pytest.raises(HTTPException): api.get_correction(old['correction_id'],OTHER)
    with pytest.raises(HTTPException):
        api._validate_and_save(OTHER,AID,image,mask,json.dumps(state),old['correction_id'])
    new=api._validate_and_save(OWNER,AID,image,mask,json.dumps(state),old['correction_id'])
    assert new['parent_id']==old['correction_id']

@pytest.mark.parametrize('field,value', [('version',2),('width',64),('closed',False),
    ('coordinates','pixels'),('points',[{'x':float('nan'),'y':0}]*3),('points',[{'x':2,'y':0}]*3)])
def test_invalid_editor(field,value,payload):
    state=payload[2]; state[field]=value
    with pytest.raises(HTTPException): validate_editor(json.dumps(state),48,64)

def test_ordinary_user_cannot_moderate(storage,monkeypatch):
    app=FastAPI(); app.include_router(api.router)
    app.dependency_overrides[api.current_user]=lambda:OWNER
    monkeypatch.setattr(api,'_config',lambda:('https://example.invalid',{},'private'))
    class Response:
        status_code=200
        def json(self): return [{'role':'user'}]
    monkeypatch.setattr(api.requests,'get',lambda *a,**k:Response())
    client=TestClient(app)
    assert client.get('/v4/corrections/workflow/queue').status_code==403
    assert client.post('/v4/corrections/workflow/review/'+OWNER+'/id',json={'decision':'accepted'}).status_code==403

def test_old_client_still_saves_and_new_route_validates(storage,payload):
    image,mask,state=payload
    app=FastAPI(); app.include_router(api.router); app.dependency_overrides[api.current_user]=lambda:OWNER
    client=TestClient(app)
    files={'image':('image.jpg',image),'mask':('mask.png',mask)}
    assert client.post('/v4/corrections',data={'analysis_id':AID},files=files).json()['saved']
    assert client.post('/v4/corrections/workflow',data={'analysis_id':AID,'editor_state':json.dumps(state)},files=files).json()['workflow_version']==1
    state['width']=2
    assert client.post('/v4/corrections/workflow',data={'analysis_id':AID,'editor_state':json.dumps(state)},files=files).status_code==422

def test_store_maps_transaction_conflict_without_leaking_response():
    class Response:
        status_code=409
        def json(self): return {'code':'23505','message':'private database details'}
    with patch('correction_workflow.requests.request',return_value=Response()):
        with pytest.raises(HTTPException) as caught:
            WorkflowStore(lambda:('https://example.invalid',{},'private')).transition('register',OWNER,'id')
    assert caught.value.status_code==409
    assert 'private' not in caught.value.detail
