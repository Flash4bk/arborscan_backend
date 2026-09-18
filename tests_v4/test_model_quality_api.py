from unittest.mock import patch
import numpy as np
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from arborscan_v4 import model_quality_api as q
from arborscan_v4.plantnet_service import PlantNetClient


def test_all_admin_routes_reject_missing_auth():
    app=FastAPI();app.include_router(q.router);client=TestClient(app)
    for method,path,body in [('get','/status',None),('get','/data',None),('get','/labels/a/b',None),
        ('post','/labels',{}),('post','/snapshots',{}),('post','/jobs',{}),
        ('post','/jobs/a/cancel',{}),('post','/activate',{})]:
        response=getattr(client,method)('/v4/model-quality'+path,**({'json':body} if body is not None else {}))
        assert response.status_code==401


def test_routes_share_server_admin_dependency():
    # Prevent accidentally introducing an unguarded registry or data route.
    for route in q.router.routes:
        assert any(d.call is q.current_admin for d in route.dependant.dependencies)


def test_plantnet_failure_never_exposes_key_or_invents_species():
    client=PlantNetClient();client.api_key='private-test-key'
    with patch('arborscan_v4.plantnet_service.requests.post',side_effect=requests.ConnectionError('https://x?api-key=private-test-key')):
        result=client.identify(np.zeros((32,32,3),np.uint8))
    assert result['status']=='network_error' and result['scientific_name'] is None
    assert 'private-test-key' not in str(result)


@pytest.mark.parametrize('name,score,expected',[('Pinus sylvestris',.8,'ok'),('Pinus sylvestris',.001,'low_confidence'),('Pinus',.9,'ok')])
def test_plantnet_provenance_uncertainty_and_genus(name,score,expected):
    client=PlantNetClient();client.api_key='test'
    payload={'version':'2026-test','query':{'secret':'not_retained'},'results':[{'score':score,'gbif':{'id':123},
        'species':{'scientificNameWithoutAuthor':name,'commonNames':[]}}]}
    with patch('arborscan_v4.plantnet_service.requests.post') as post:
        post.return_value.status_code=200;post.return_value.json.return_value=payload
        result=client.identify(np.zeros((32,32,3),np.uint8))
    assert result['status']==expected
    assert result['engine_version']=='2026-test' and result['retrieved_at']
    assert result['score_interpretation']=='provider_ranking_score_not_measured_accuracy'
    assert 'query' not in result['original_prediction']
    if expected!='ok':assert result['scientific_name'] is None
    elif name=='Pinus':assert result['taxon_rank']=='genus_or_unresolved'
    else:assert result['taxon_id']=='123' and result['russian_name'] is None
