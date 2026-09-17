import json
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from arborscan_v4 import report_history as h

IMAGE=(Path(__file__).parents[1]/'arborscan_app/test/fixtures/reference_exif6.jpg').read_bytes()
OWNER='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'

def snapshot():
    line=lambda a,b:[{'x':a[0],'y':a[1]},{'x':b[0],'y':b[1]}]
    return {'version':1,'kind':'reference','change_source':'reference','captured_at':'2026-09-17T00:00:00Z',
      'report':{'height_m':999},'reference':{'version':1,'method':'known_object_segment_v1',
      'coordinates':'normalized_oriented_image','width':100,'height':200,'same_plane':True,'length_m':1,
      'reference':line((.1,.1),(.1,.3)),'tree':line((.5,.1),(.5,.9)),
      'crown':line((.2,.1),(.8,.1)),'outline':[{'x':.1,'y':.1},{'x':.1,'y':.3},{'x':.2,'y':.3}]}}

def test_exif_restore_reference_and_full_snapshot():
    d=snapshot(); d['environment']={'weather':{'value':{'temperature':8},'source':'test_station','retrieved_at':'2026-09-16T00:00:00Z'}}
    r=h.validate_snapshot(json.dumps(d),IMAGE)
    assert (r['image']['width'],r['image']['height'])==(100,200)
    assert r['report']['height_m']==pytest.approx(4)
    assert r['report']['crown_width_m']==pytest.approx(1.5)
    assert r['environment']==d['environment']
    assert r['report']['beta_kg_s'] is None
    assert r['reference']==d['reference']

@pytest.mark.parametrize('mutation',[
 lambda d:d['reference'].update(width=200,height=100),
 lambda d:d['reference'].update(length_m=float('nan')),
 lambda d:d['reference'].update(same_plane=False),
 lambda d:d.update(ar={'photo_sha256':'wrong'}),
 lambda d:d.update(owner_id=OWNER),
 lambda d:d.update(environment={'weather':{'value':8}}),
])
def test_invalid_binding_and_provenance(mutation):
    d=snapshot();mutation(d)
    with pytest.raises(HTTPException) as e:h.validate_snapshot(json.dumps(d),IMAGE)
    assert e.value.status_code==422

def test_files_require_owned_index_not_only_known_hash():
    store=object.__new__(h.ReportStore)
    with patch.object(store,'request',return_value=[]) as req,patch.object(store,'read_blob') as blob:
        with pytest.raises(HTTPException) as e:store.get(OWNER,'11111111-1111-4111-8111-111111111111')
        assert e.value.status_code==404
        assert req.call_args.kwargs['params']['owner_id']=='eq.'+OWNER
        blob.assert_not_called()

def test_partial_upload_never_registers_or_reports_saved():
    store=object.__new__(h.ReportStore)
    payload=h.validate_snapshot(json.dumps(snapshot()),IMAGE)
    with patch.object(store,'ready'),patch.object(store,'write_blob',side_effect=HTTPException(503,'partial')),patch.object(store,'request') as rpc:
        with pytest.raises(HTTPException):store.save(OWNER,OWNER,OWNER,None,payload)
        rpc.assert_not_called()

def test_all_routes_require_authorization():
    app=FastAPI();app.include_router(h.router)
    client=TestClient(app)
    for url in ['/v4/reports','/v4/reports/capabilities','/v4/reports/'+OWNER]:
        assert client.get(url).status_code==401

def test_legacy_snapshot_does_not_invent_reference_or_modern_provenance():
    d={'version':1,'kind':'legacy','change_source':'legacy_import','captured_at':'2026-01-01T00:00:00Z','report':{'species':'Pine'}}
    r=h.validate_snapshot(json.dumps(d),IMAGE)
    assert 'reference' not in r and 'height_m' not in r['report']
    assert r['provenance']=='user_supplied_snapshot_not_independently_verified'
