import base64
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_selfhost_audit as audit
import ops_selfhost_public_check as public


class PublicRecoveryChecks(unittest.TestCase):
    def test_network_check_keeps_tls_and_original_public_prefixes(self):
        client = public.PublicHTTP()
        class Response:
            status_code = 401
            def json(self):
                return {'detail': 'Unauthorized'}
        class Session:
            def request(self, method, url, **kwargs):
                self.method, self.url, self.kwargs = method, url, kwargs
                return Response()
        session = Session()
        client.session = session
        client.api('GET', '/api/v4/v4/reports', expected=401)
        self.assertEqual(session.url, 'https://31.57.170.88/api/v4/v4/reports')
        self.assertIs(session.kwargs['verify'], True)
        self.assertIs(session.kwargs['allow_redirects'], False)
        self.assertEqual(session.kwargs['headers'], {})
        with self.assertRaises(audit.AuditFailure):
            client.api('GET', '/storage/v1/object/private')

    def test_plain_server_error_is_failure_not_successful_login(self):
        client = public.PublicHTTP()
        class Response:
            status_code = 500
            def json(self):
                raise ValueError('Internal Server Error')
        class Session:
            def request(self, *args, **kwargs):
                return Response()
        client.session = Session()
        with self.assertRaisesRegex(audit.AuditFailure, 'public_http_status_mismatch'):
            client.api('POST', '/api/v3/auth/login')
        Response.status_code = 200
        with self.assertRaisesRegex(audit.AuditFailure, 'public_json_required'):
            client.api('POST', '/api/v3/auth/login')

    def test_report_readback_rejects_wrong_photo_mask_values_and_unconfirmed_save(self):
        encoded = base64.b64encode(b'original-photo').decode()
        state = {'image': encoded, 'snapshot': {'report': {
            'measurements': {'height_m': 4, 'dbh_cm': None}, 'images': {'mask': 'mask-a'}}}}
        value = {'persisted': True, 'snapshot': {
            'image': {'original_base64': encoded}, 'report': copy.deepcopy(state['snapshot']['report'])}}
        public.check_report(value, state)
        cases = []
        wrong = copy.deepcopy(value); wrong['persisted'] = False; cases.append(wrong)
        wrong = copy.deepcopy(value); wrong['snapshot']['image']['original_base64'] = base64.b64encode(b'other').decode(); cases.append(wrong)
        wrong = copy.deepcopy(value); wrong['snapshot']['report']['measurements']['height_m'] = 400; cases.append(wrong)
        wrong = copy.deepcopy(value); wrong['snapshot']['report']['images']['mask'] = 'mask-b'; cases.append(wrong)
        for changed in cases:
            with self.subTest(value=changed), self.assertRaises(audit.AuditFailure):
                public.check_report(changed, state)
