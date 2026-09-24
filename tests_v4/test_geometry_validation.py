import pytest
from tools.geometry_validation import summarize


def row(**kw):
    return dict(object_id='synthetic',experiment_id='fixture',app_version='test',algorithm_version='test',
        method='reference',metric='height',unit='m',control_value='10',control_method='synthetic',
        repeat='1',result='11',conditions='fixed',**kw)


def test_errors_repeatability_and_missing_uncertainty():
    a=row();b={**a,'repeat':'2','result':'9'}
    r=summarize([a,b]);g=r['groups'][0]
    assert g['bias']==0 and g['mae']==1 and g['rmse']==1
    assert g['repeatability_sample_sd']==pytest.approx(2**.5)
    assert r['observations'][0]['control_uncertainty'] is None
    assert r['observations'][1]['relative_error']==-.1


def test_zero_and_missing_control_and_groups_do_not_mix():
    a=row();a['control_value']='0'
    b={**a,'experiment_id':'other','control_value':'','result':''}
    r=summarize([a,b]);assert len(r['groups'])==2
    assert r['observations'][0]['relative_error'] is None
    assert r['groups'][1]['bias'] is None
    assert r['groups'][0]['repeatability_sample_sd'] is None
    with pytest.raises(ValueError): summarize([a,a])
    with pytest.raises(ValueError): summarize([{**a,'result':'NaN'}])
