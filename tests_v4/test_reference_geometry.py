import copy
import json
import math
import pytest
from fastapi import HTTPException
from tests_v4.test_report_history import IMAGE, snapshot
from arborscan_v4.report_history import validate_snapshot


def v2():
    d=snapshot(); r=d['reference']
    r.update(version=2,method='known_object_segment_v2',scale_origin='vertical_reference_same_depth_user_confirmed',
        crown_height=[{'x':.5,'y':.4},{'x':.5,'y':.1}],
        trunk=[{'x':.45,'y':.7},{'x':.55,'y':.7}],
        trunk_axis=[{'x':.5,'y':.8},{'x':.5,'y':.6}])
    return d


def test_extended_geometry_server_recomputes_and_keeps_original_points():
    d=v2();d['report']['geometry']={'crown_height':{'value':999}}
    result=validate_snapshot(json.dumps(d),IMAGE)
    g=result['report']['geometry']
    assert g['crown_height']['value']==pytest.approx(1.5)
    assert g['trunk_diameter']['value']==pytest.approx(.25)
    assert g['trunk_lean']['value']==0
    assert g['dbh']['value'] is None and g['crown_porosity']['value'] is None
    assert result['reference']==d['reference']
    assert d['report']['geometry']['crown_height']['value']==999
    # Cold read/serialize, no in-memory dependency.
    restored=json.loads(json.dumps(result))
    assert restored['report']['geometry']==g


def test_roll_and_scale_and_height_not_segment_length():
    d=v2();d['reference']['tree'][1]['x']=.8
    first=validate_snapshot(json.dumps(d),IMAGE)['report']['geometry']
    assert first['tree_segment_length']['value']>first['tree_height']['value']
    # Rotate all original-pixel vectors while keeping the actual image size.
    from arborscan_v4.reference_geometry import reference_geometry
    r=copy.deepcopy(d['reference']); r['width'],r['height']=r['height'],r['width']
    for key in ('reference','tree','crown','outline','crown_height','trunk','trunk_axis'):
        r[key]=[{'x':1-p['y'],'y':p['x']} for p in r[key]]
    rotated=reference_geometry(r)
    for key in first:
        if first[key]['value'] is not None: assert rotated[key]['value']==pytest.approx(first[key]['value'])
    r['length_m']*=2
    scaled=reference_geometry(r)
    assert scaled['trunk_diameter']['value']==pytest.approx(first['trunk_diameter']['value']*2)
    assert scaled['trunk_lean']['value']==first['trunk_lean']['value']


@pytest.mark.parametrize('field,value',[
 ('trunk_axis',[{'x':.5,'y':.5}]*2),('trunk',[{'x':2,'y':.5}]*2),
 ('crown_height',[{'x':.1,'y':.5},{'x':.8,'y':.5}]),
 ('trunk_axis',[{'x':True,'y':.2},{'x':.5,'y':.5}]),
 ('trunk',[{'x':.45,'y':.95},{'x':.55,'y':.95}]),
 ('scale_origin','ar'),('version',True)])
def test_reject_ambiguous_or_unbound_geometry(field,value):
    d=v2();d['reference'][field]=value
    with pytest.raises(HTTPException) as e: validate_snapshot(json.dumps(d),IMAGE)
    assert e.value.status_code==422


def test_missing_optional_points_and_old_method_are_not_filled():
    d=v2()
    for key in ('trunk','trunk_axis','crown_height'): d['reference'][key]=[]
    g=validate_snapshot(json.dumps(d),IMAGE)['report']['geometry']
    assert g['trunk_diameter']['value'] is None
    old=validate_snapshot(json.dumps(snapshot()),IMAGE)
    assert old['report']['method']=='known_object_segment_v1'
    assert 'geometry' not in old['report']
