"""Multipart contract tests with deterministic vision, no database/network writes.

Run in the API image (ultralytics required). Local lightweight env may skip.
"""
from io import BytesIO
import hashlib
import pytest
import numpy as np
from PIL import Image
pytest.importorskip('ultralytics')
from fastapi.testclient import TestClient
from arborscan_v4 import api
from arborscan_v4.vision_engine import DetectionResult


@pytest.fixture
def client(monkeypatch):
    def infer(image, *_):
        h,w=image.shape[:2]
        mask=np.zeros((h,w),dtype=np.uint8);mask[10:h-10,10:w-10]=255
        return DetectionResult(detected=True,mask=mask,confidence=.9,
                               bbox=(10,10,w-10,h-10),class_id=0,class_name='tree')
    monkeypatch.setattr(api.vision,'infer',infer)
    monkeypatch.setattr(api.plantnet,'identify',lambda _: {'status':'not_run','display_name':'Test'})
    return TestClient(api.app)


def photo():
    im=Image.new('RGB',(200,100),'white');exif=im.getexif();exif[274]=6
    b=BytesIO();im.save(b,format='JPEG',exif=exif);return b.getvalue()


def post(client, **fields):
    return client.post('/v4/analyze-tree',files={'file':('synthetic.jpg',photo(),'image/jpeg')},
                       data={'include_images':'false',**fields})


def test_reference_si_and_oriented_original_pixels(client):
    r=post(client,reference_length_m='1',reference_length_px='20',
           reference_same_plane='true',crown_width_px='60')
    assert r.status_code==200,r.text
    j=r.json()
    assert j['measurement_method_version']==2
    assert j['geometry']['dbh']['value'] is None
    assert j['geometry']['crown_porosity']['value'] is None
    assert j['pixel_measurements']['image_width_px']==100
    assert j['pixel_measurements']['image_height_px']==200
    assert j['measurements']['height']['value_m']==9
    assert j['measurements']['crown_width']['value_m']==3
    assert j['measurements']['trunk_diameter']['value_m'] is None
    assert j['persisted'] is False and j['beta']['value_kg_s'] is None
    assert post(client,reference_length_m='1',reference_length_px='20',
                reference_same_plane='false').json()['calibration']['available'] is False


def test_ar_raw_photo_hash_confirmation_and_old_client(client):
    fields={'ar_height_m':'12','ar_trunk_diameter_m':'.3','ar_trunk_measurement_height_m':'1.4',
            'ar_same_tree_confirmed':'true','ar_photo_sha256':hashlib.sha256(photo()).hexdigest()}
    j=post(client,**fields).json()
    assert j['measurements']['height']['value_m']==12
    assert j['measurements']['trunk_diameter']['standard'] is None
    assert j['measurements']['crown_width']['value_m'] is None
    assert j['calibration']['available'] is False
    for changes in [{'ar_photo_sha256':'0'*64},{'ar_same_tree_confirmed':'false'}]:
        j=post(client,**{**fields,**changes}).json()
        assert j['measurements']['height']['value_m'] is None
    assert post(client,ar_height_m='12').json()['measurements']['height']['value_m'] is None


def test_invalid_numbers_rejected_and_contours_still_private(client):
    for field in ['reference_length_m','reference_length_px','ar_height_m','crown_width_px']:
        for value in ['NaN','Infinity','0','-1']:
            assert post(client,**{field:value}).status_code==422
    assert client.get('/v4/corrections').status_code==401
