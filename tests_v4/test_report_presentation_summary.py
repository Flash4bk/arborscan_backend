from copy import deepcopy
from unittest.mock import patch

from arborscan_v4 import report_history as h

OWNER = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'


def saved_summary(report, **extra):
    store = object.__new__(h.ReportStore)
    payload = {
        'kind': 'v4', 'report': report,
        'captured_at': '2026-11-12T00:00:00Z',
        'image': {'sha256': 'demo-image-hash'}, **extra,
    }
    original = deepcopy(payload)
    committed = {'owner_id': OWNER, 'analysis_id': OWNER, 'version_id': OWNER,
                 'parent_id': None, 'payload_sha256': 'demo-payload', 'image_sha256': 'demo-image-hash'}
    with patch.object(store, 'ready'), patch.object(store, 'write_blob', return_value='demo-payload'), patch.object(store, 'request', return_value=committed) as request:
        store.save(OWNER, OWNER, OWNER, None, payload)
    assert payload == original
    return request.call_args.kwargs['json']['p_summary']


def test_manual_species_summary_retains_original_without_confirming_label():
    summary = saved_summary({
        'species': {'display_name': 'DEMO original', 'status': 'confirmed'},
        'species_annotation': {
            'version': 1, 'display_name': 'DEMO manual',
            'source': 'user', 'confirmed': False,
        },
    })
    assert summary['species'] == 'DEMO manual'
    assert summary['inference_species'] == 'DEMO original'
    assert summary['species_source'] == 'user'
    assert summary['species_confirmed'] is False
    assert summary['presentation_version'] == 1


def test_summary_prefers_selected_reference_section_over_old_ar_measurement():
    summary = saved_summary({
        'height_m': 12, 'crown_width_m': 4,
        'measurements': {'trunk_diameter': {'value_m': .4, 'source': 'ar'}},
        'geometry': {'trunk_diameter': {
            'value': .25, 'unit': 'm', 'source': 'reference', 'available': True,
        }},
    }, kind='reference', reference={'version': 2, 'method': 'known_object_segment_v2'})
    assert summary['height_m'] == 12
    assert summary['crown_width_m'] == 4
    assert summary['trunk_diameter_m'] == .25


def test_summary_never_falls_back_over_explicitly_unavailable_metric():
    summary = saved_summary({
        'height_m': 77,
        'measurements': {'height': {'value_m': 99, 'available': False, 'source': 'unavailable'}},
    })
    assert summary['height_m'] is None


def test_old_summary_is_labelled_without_blob_reads_or_rewriting_stored_rows():
    store = object.__new__(h.ReportStore)
    rows = [{'version_id': OWNER, 'owner_id': OWNER, 'summary': {'species': 'DEMO legacy'}}]
    original = deepcopy(rows)
    with patch.object(store, 'request', return_value=rows), patch.object(store, 'read_blob') as read:
        result = store.listing(OWNER, 0)
    read.assert_not_called()
    assert rows == original
    assert result['items'][0]['summary']['species'] == 'DEMO legacy'
    assert result['items'][0]['summary']['presentation_version'] == 0
    assert result['items'][0]['summary']['species_source'] == 'legacy_inference'


def test_malformed_optional_maps_never_turn_snapshot_save_into_500():
    for value in ([], 'old-format', 42):
        report = {'height_m': 12, 'measurements': value, 'measurement_sources': value,
                  'geometry': value}
        summary = saved_summary(report)
        assert summary['height_m'] == 12
        summary = saved_summary(report, kind='reference', reference={'version': 1})
        assert summary['trunk_diameter_m'] is None
    summary = saved_summary({'geometry': {'trunk_diameter': [1, 2]}},
                            kind='reference', reference={'version': 1})
    assert summary['trunk_diameter_m'] is None
