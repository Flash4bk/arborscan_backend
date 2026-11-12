"""Science transport regressions with an isolated, durable fake index.

These do not claim a production database migration or field validation.
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

OWNER = '11111111-1111-4111-8111-111111111111'
OTHER = '22222222-2222-4222-8222-222222222222'
ANALYSIS = '33333333-3333-4333-8333-333333333333'
SOURCE = '44444444-4444-4444-8444-444444444444'
RESULT = '55555555-5555-4555-8555-555555555555'
IMAGE = 'a' * 64


def inputs():
    return {
        'schema_version': 1,
        'section_model': 'healthy_solid_circular',
        'strength_definition': 'Synthetic effective strength, no added modifiers.',
        'diameter': {'value': .2, 'unit': 'm', 'origin': 'scenario_assumption', 'source': 'Synthetic test'},
        'effective_mor': {'value': 50e6, 'unit': 'Pa', 'origin': 'scenario_assumption', 'source': 'Synthetic test'},
    }


class DiskStore:
    """A new instance rereads files; no process-memory persistence assumption."""
    def __init__(self, directory):
        self.path = Path(directory) / 'index.json'

    def rows(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def seed(self):
        payload = {'version': 1, 'kind': 'v4', 'report': {'height_m': 12.0, 'beta_kg_s': None},
                   'captured_at': '2026-11-12T00:00:00Z', 'change_source': 'analysis',
                   'image': {'sha256': IMAGE, 'original_base64': 'synthetic-photo'}}
        self.save(OWNER, ANALYSIS, SOURCE, None, payload)

    def get(self, owner, version):
        data = self.rows().get(owner + '/' + version)
        if data is None:
            raise HTTPException(404, 'Report not found')
        return copy.deepcopy(data)

    def save(self, owner, analysis, version, parent, payload):
        rows = self.rows()
        key = owner + '/' + version
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        record = {'owner_id': owner, 'analysis_id': analysis, 'version_id': version,
                  'parent_id': parent, 'payload_sha256': digest, 'image_sha256': IMAGE}
        if key in rows:
            if rows[key]['record'] != record:
                raise HTTPException(409, 'Idempotency conflict')
            return {'saved': True, 'persisted': True, 'record': record}
        if parent and any(x['record']['owner_id'] == owner and x['record']['parent_id'] == parent
                          for x in rows.values()):
            raise HTTPException(409, 'Revision conflict')
        rows[key] = {'record': record, 'snapshot': payload, 'persisted': True}
        self.path.write_text(json.dumps(rows))
        return {'saved': True, 'persisted': True, 'record': record}


class ScienceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = DiskStore(self.tmp.name)
        self.store.seed()

    def call(self, owner=OWNER, values=None, version=RESULT):
        from arborscan_v4.science_api import save_section_result
        return save_section_result(owner, SOURCE, version, values or inputs(), self.store)

    def test_server_calculates_immutable_child_and_reopens_after_service_recreation(self):
        original = self.store.get(OWNER, SOURCE)
        saved = self.call()
        self.assertIs(saved['saved'], True)
        reopened = DiskStore(self.tmp.name).get(OWNER, RESULT)
        result = reopened['snapshot']['science_result']
        self.assertEqual(result['source_version_id'], SOURCE)
        self.assertEqual(result['source_payload_sha256'], original['record']['payload_sha256'])
        self.assertEqual(result['image_sha256'], IMAGE)
        self.assertEqual(result['capacity_nm'], 39269.90816987242)
        self.assertEqual(len(result['implementation_sha256']), 64)
        self.assertFalse(result['field_validated'])
        self.assertIsNone(result['beta_kg_s'])
        self.assertEqual(self.store.get(OWNER, SOURCE), original)
        self.assertEqual(reopened['snapshot']['report'], original['snapshot']['report'])

    def test_identical_retry_after_lost_response_is_idempotent(self):
        self.call()
        self.store = DiskStore(self.tmp.name)
        again = self.call()
        self.assertTrue(again['persisted'])
        self.assertEqual(len(self.store.rows()), 2)

    def test_committed_retry_survives_implementation_upgrade_without_recomputation(self):
        from unittest.mock import patch
        first = self.call()
        with patch('arborscan_v4.science_api.section_capacity', side_effect=AssertionError('must replay committed result')):
            again = self.call()
        self.assertEqual(again['science_result'], first['science_result'])
        self.assertEqual(len(self.store.rows()), 2)

    def test_changed_inputs_same_operation_and_concurrent_child_conflict(self):
        self.call()
        changed = inputs()
        changed['diameter']['value'] = .25
        for version in (RESULT, '66666666-6666-4666-8666-666666666666'):
            with self.assertRaises(HTTPException) as error:
                self.call(values=changed, version=version)
            self.assertEqual(error.exception.status_code, 409)

    def test_other_owner_cannot_read_or_append(self):
        with self.assertRaises(HTTPException) as error:
            self.call(owner=OTHER)
        self.assertEqual(error.exception.status_code, 404)
        self.assertEqual(len(self.store.rows()), 1)

    def test_client_cannot_supply_result_beta_or_invented_interval(self):
        for key, value in [('beta_kg_s', 44.4), ('capacity_nm', 123), ('uncertainty_interval', [1, 2])]:
            changed = {**inputs(), key: value}
            with self.assertRaises(HTTPException) as error:
                self.call(values=changed)
            self.assertEqual(error.exception.status_code, 422)
        self.assertEqual(len(self.store.rows()), 1)

    def test_nonconfirmed_save_is_not_success(self):
        self.store.save = lambda *a: {'saved': False, 'persisted': False}
        with self.assertRaises(HTTPException) as error:
            self.call()
        self.assertEqual(error.exception.status_code, 503)

    def test_routes_require_authentication(self):
        from arborscan_v4.science_api import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        self.assertEqual(client.get('/v4/science/capabilities').status_code, 401)
        self.assertEqual(client.post('/v4/science/section-capacity', json={}).status_code, 401)

    def test_existing_report_store_rejects_empty_or_mismatched_commit_reply(self):
        from unittest.mock import patch
        from arborscan_v4.report_history import ReportStore
        store = object.__new__(ReportStore)
        payload = self.store.get(OWNER, SOURCE)['snapshot']
        for reply in (None, {}, {'version_id': OTHER},
                      {'version_id': RESULT, 'owner_id': OTHER}):
            with patch.object(store, 'ready'), patch.object(store, 'write_blob', return_value='b'*64), \
                    patch.object(store, 'request', return_value=reply):
                with self.assertRaises(HTTPException) as error:
                    store.save(OWNER, ANALYSIS, RESULT, SOURCE, payload)
                self.assertEqual(error.exception.status_code, 503)

    def test_snapshot_upload_cannot_forge_trusted_scientific_field(self):
        from tests_v4.test_report_history import snapshot, IMAGE as PHOTO
        from arborscan_v4.report_history import validate_snapshot
        forged = {**snapshot(), 'science_result': {'beta_kg_s': 44.4, 'field_validated': True}}
        with self.assertRaises(HTTPException) as error:
            validate_snapshot(json.dumps(forged), PHOTO)
        self.assertEqual(error.exception.status_code, 422)


if __name__ == '__main__':
    unittest.main()
