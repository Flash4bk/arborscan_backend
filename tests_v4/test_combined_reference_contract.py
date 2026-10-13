"""Existing production snapshot contract accepts AI + manual reference together.

No HTTP, production records or trained weights are needed for these checks.
"""
import copy
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from arborscan_v4.report_history import validate_snapshot

PHOTO = (Path(__file__).parents[1] / 'arborscan_app/test/fixtures/reference_exif6.jpg').read_bytes()


def combined():
    return {
        'version': 1, 'kind': 'reference', 'change_source': 'reference',
        'captured_at': '2026-10-13T10:00:00Z',
        'report': {
            'analysis_id': 'DEMO-combined',
            'tree': {'detected': True},
            'species': {'display_name': 'DEMO'},
            'measurements': {'height': {'value_m': None, 'source': 'unavailable',
                                       'notes': ['absolute_scale_required']}},
            'images': {'width': 100, 'height': 200, 'mask_image_base64': 'DEMO-saved-preview'},
        },
        'reference': {
            'version': 1, 'method': 'known_object_segment_v1',
            'coordinates': 'normalized_oriented_image',
            'width': 100, 'height': 200, 'same_plane': True, 'length_m': 1,
            'reference': [{'x': .1, 'y': .8}, {'x': .1, 'y': .6}],
            'tree': [{'x': .5, 'y': .8}, {'x': .5, 'y': .1}],
            'crown': [{'x': .2, 'y': .2}, {'x': .8, 'y': .2}],
            'outline': [{'x': .08, 'y': .8}, {'x': .12, 'y': .8}, {'x': .1, 'y': .6}],
        },
    }


def test_combined_snapshot_computes_reference_without_overwriting_model_measurements():
    request = combined()
    saved = validate_snapshot(json.dumps(request), PHOTO)
    assert saved['report']['measurements'] == request['report']['measurements']
    assert saved['report']['images'] == request['report']['images']
    assert saved['report']['species'] == request['report']['species']
    assert saved['report']['height_m'] == pytest.approx(3.5)
    assert saved['report']['crown_width_m'] == pytest.approx(1.5)
    assert saved['image']['width'] == 100 and saved['image']['height'] == 200
    assert saved['provenance'] == 'user_supplied_snapshot_not_independently_verified'
    assert saved['report']['beta_kg_s'] is None
    # Scaling a new version does not mutate the earlier accepted payload.
    original = copy.deepcopy(saved)
    request['reference']['length_m'] = 2
    revised = validate_snapshot(json.dumps(request), PHOTO)
    assert revised['report']['height_m'] == pytest.approx(7)
    assert saved == original


def test_failed_segmentation_still_allows_explicit_manual_reference():
    request = combined()
    request['report']['tree']['detected'] = False
    request['report']['images'].pop('mask_image_base64')
    saved = validate_snapshot(json.dumps(request), PHOTO)
    assert saved['report']['height_m'] == pytest.approx(3.5)
    assert saved['report']['tree']['detected'] is False
    assert 'mask_image_base64' not in saved['report']['images']


def test_original_exif_coordinate_binding_is_required_even_with_model_preview():
    request = combined()
    request['report']['images'].update(width=200, height=100)
    with pytest.raises(HTTPException) as exc:
        validate_snapshot(json.dumps(request), PHOTO)
    assert exc.value.status_code == 422
