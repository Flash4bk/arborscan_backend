import asyncio
import base64
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arborscan_v4"))
import corrections_api as api

OWNER = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
OTHER = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
ANALYSIS = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imencode('.jpg', np.full((64, 48, 3), 150, np.uint8))[1].tobytes()
        self.mask_array = np.zeros((64, 48), np.uint8)
        self.mask_array[5:55, 10:30] = 255
        self.mask = cv2.imencode('.png', self.mask_array)[1].tobytes()
        self.records = {}
        def put(owner, key, record):
            return self.records.setdefault((owner, key), record)
        self.patcher = patch.object(api, '_put', side_effect=put)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def save(self, mask=None, owner=OWNER):
        return api._validate_and_save(owner, ANALYSIS, self.image, mask or self.mask)

    def test_roundtrip_preserves_bytes_pending_review(self):
        result = self.save()
        record = self.records[(OWNER, result['correction_id'])]
        self.assertTrue(result['saved'])
        self.assertEqual(base64.b64decode(record['original_image_base64']), self.image)
        self.assertEqual(base64.b64decode(record['mask_png_base64']), self.mask)
        self.assertEqual(record['review_status'], 'pending_review')
        self.assertFalse(record['eligible_for_training'])

    def test_retry_keeps_same_revision(self):
        self.assertEqual(self.save(), self.save())
        self.assertEqual(len(self.records), 1)

    def test_changed_mask_creates_new_revision(self):
        first = self.save()
        self.mask_array[5:55, 30] = 255
        second = self.save(cv2.imencode('.png', self.mask_array)[1].tobytes())
        self.assertNotEqual(first['revision'], second['revision'])
        self.assertEqual(len(self.records), 2)

    def test_owner_namespaces_are_separate(self):
        first = self.save()
        self.save(owner=OTHER)
        self.assertIn((OWNER, first['correction_id']), self.records)
        self.assertIn((OTHER, first['correction_id']), self.records)

    def test_rejects_mismatched_dimensions(self):
        with self.assertRaises(HTTPException) as caught:
            self.save(cv2.imencode('.png', self.mask_array[:32])[1].tobytes())
        self.assertEqual(caught.exception.status_code, 422)
        self.assertFalse(self.records)

    def test_rejects_empty_full_and_non_png_masks(self):
        for array, ext in [(np.zeros((64,48),np.uint8), '.png'),
                           (np.full((64,48),255,np.uint8), '.png'),
                           (self.mask_array, '.jpg')]:
            with self.subTest(ext=ext, pixels=int(array.sum())):
                with self.assertRaises(HTTPException):
                    self.save(cv2.imencode(ext, array)[1].tobytes())
        self.assertFalse(self.records)

    def test_missing_authorization_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            api.current_user(None)
        self.assertEqual(caught.exception.status_code, 401)

    def test_object_paths_are_scoped_to_owner(self):
        key = ANALYSIS + '_' + 'a'*64 + '.json'
        with patch.object(api, '_config', return_value=('https://example.invalid', {}, 'raw')):
            url, _ = api._object_url(OWNER, key)
            other_url, _ = api._object_url(OTHER, key)
            self.assertIn('/' + OWNER + '/', url)
            self.assertNotEqual(url, other_url)
            with self.assertRaises(HTTPException): api._object_url(OWNER, '../escape.json')

    def test_expired_session_rejected(self):
        class Response:
            status_code = 200
            def json(self): return []
        with patch.object(api, '_config', return_value=('https://example.invalid', {}, 'raw')):
            with patch.object(api.requests, 'get', return_value=Response()) as get:
                with self.assertRaises(HTTPException) as caught:
                    api.current_user('Bearer abcdefghijklmnopqrstuvwxyz')
                self.assertEqual(caught.exception.status_code, 401)
                self.assertTrue(get.call_args.kwargs['params']['expires_at'].startswith('gt.'))

    def test_upload_size_limit(self):
        with patch.object(api, 'MAX_IMAGE', 4):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(api.save_correction(analysis_id=ANALYSIS,
                    image=UploadFile(io.BytesIO(self.image)),
                    mask=UploadFile(io.BytesIO(self.mask)), owner=OWNER))
        self.assertEqual(caught.exception.status_code, 413)
        self.assertFalse(self.records)


if __name__ == '__main__':
    unittest.main()
