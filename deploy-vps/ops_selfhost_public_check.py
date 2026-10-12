"""Exercise the real public HTTPS API with this recovery's DEMO accounts only.

No TLS exceptions, service-role credentials, real-user logins, moderation
changes or training calls. Private retry state makes report saves idempotent.
"""
import argparse
import base64
import datetime
import json
from pathlib import Path
import sys
import uuid

import ops_selfhost_audit as audit

BASE = 'https://31.57.170.88'


class PublicHTTP:
    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False

    def api(self, method, path, token=None, expected=200, **kwargs):
        if not (path.startswith('/api/v3/') or path.startswith('/api/v4/')):
            raise audit.AuditFailure('public_api_route_required')
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        response = self.session.request(method, BASE + path, headers=headers,
                                        timeout=(15, 240), allow_redirects=False,
                                        verify=True, **kwargs)
        if response.status_code != expected:
            raise audit.AuditFailure('public_http_status_mismatch')
        try:
            result = response.json()
        except ValueError:
            raise audit.AuditFailure('public_json_required') from None
        if not isinstance(result, dict):
            raise audit.AuditFailure('public_object_required')
        return result


def check_report(actual, state):
    snapshot = actual.get('snapshot', {})
    if (actual.get('persisted') is not True
            or base64.b64decode(snapshot['image']['original_base64'], validate=True)
            != base64.b64decode(state['image'], validate=True)
            or snapshot['report']['measurements'] != state['snapshot']['report']['measurements']
            or snapshot['report']['images'] != state['snapshot']['report']['images']):
        raise audit.AuditFailure('public_photo_mask_measurements_mismatch')


def check(root):
    root = Path(root)
    if (not (root / 'PROMOTED').is_file()
            or audit.private_json(root / 'promotion.private.json').get('phase') != 'promoted'):
        raise audit.AuditFailure('actual_promotion_required')
    fixture = audit.validate_fixture(audit.private_json(root / 'fixture.private.json'))
    if fixture.get('complete') is not True:
        raise audit.AuditFailure('complete_demo_fixture_required')
    owner, other = fixture['accounts']
    http = PublicHTTP()
    for path in ('/api/v3/health', '/api/v4/health'):
        if http.api('GET', path).get('status') != 'ok':
            raise audit.AuditFailure('public_health_mismatch')
    for path in ('/api/v3/auth/me', '/api/v4/v4/reports', '/api/v4/v4/corrections'):
        http.api('GET', path, expected=401)
    http.api('POST', '/api/v3/auth/login', expected=401,
             json={'email': owner['email'], 'password': 'DEMO-incorrect-password'})
    tokens = []
    for account in (owner, other):
        login = http.api('POST', '/api/v3/auth/login',
                         json={key: account[key] for key in ('email', 'password')})
        if login['user']['id'] != account['id'] or not login.get('token'):
            raise audit.AuditFailure('public_demo_owner_mismatch')
        me = http.api('GET', '/api/v3/auth/me', login['token'])
        if me.get('ok') is not True or me['user']['id'] != account['id']:
            raise audit.AuditFailure('public_session_owner_mismatch')
        tokens.append(login['token'])
    token, second = tokens
    http.api('GET', '/api/v4/v4/corrections/workflow/queue', token, 403)
    if (http.api('GET', '/api/v4/v4/reports/capabilities', token)['history_version'] != 1
            or http.api('GET', '/api/v4/v4/corrections/workflow/capabilities', token)['workflow_version'] != 1):
        raise audit.AuditFailure('public_capabilities_mismatch')
    for version in fixture['versions']:
        report = http.api('GET', '/api/v4/v4/reports/' + version, token)
        if (report.get('persisted') is not True
                or report['snapshot']['image']['sha256'] != fixture['image_sha256']
                or abs(report['snapshot']['report']['height_m'] - 4) > 1e-8):
            raise audit.AuditFailure('public_reference_versions_mismatch')
        http.api('GET', '/api/v4/v4/reports/' + version, second, 404)
    for correction, status in fixture['contours']:
        value = http.api('GET', '/api/v4/v4/corrections/' + correction, token)
        if value['review_status'] != status or value['image_sha256'] != fixture['image_sha256']:
            raise audit.AuditFailure('public_contour_status_or_photo_mismatch')
        http.api('GET', '/api/v4/v4/corrections/' + correction, second, 404)
    previous = audit.private_json(root / 'inference.fixture.private.json')
    check_report(http.api('GET', '/api/v4/v4/reports/' + previous['version'], token), previous)

    path = root / 'public-inference.fixture.private.json'
    if path.exists():
        state = audit.private_json(path)
        if state.get('owner') != owner['id'] or state.get('tag') != audit.FIXTURE_NAME:
            raise audit.AuditFailure('public_retry_owner_mismatch')
    else:
        raw = base64.b64decode(previous['image'], validate=True)
        result = http.api('POST', '/api/v4/v4/analyze-tree', token,
                          files={'file': ('DEMO-public-inference.jpg', raw, 'image/jpeg')},
                          data={'tap_x': '.5', 'tap_y': '.4'})
        if (result.get('tree', {}).get('detected') is not True
                or result.get('segmentation_weights_sha256')
                != previous['snapshot']['report']['segmentation_weights_sha256']):
            raise audit.AuditFailure('public_inference_model_mismatch')
        state = {'owner': owner['id'], 'tag': audit.FIXTURE_NAME,
                 'image': previous['image'], 'analysis': result['analysis_id'],
                 'version': str(uuid.uuid4()), 'snapshot': {'version': 1, 'kind': 'v4',
                 'change_source': 'analysis',
                 'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'report': result}}
        audit.write_private(path, state)
    data = {'analysis_id': state['analysis'], 'version_id': state['version'],
            'snapshot': json.dumps(state['snapshot'])}
    files = {'image': ('DEMO-public-inference.jpg', base64.b64decode(state['image'], validate=True), 'image/jpeg')}
    saved = http.api('POST', '/api/v4/v4/reports', token, data=data, files=files)
    if saved.get('saved') is not True or saved.get('persisted') is not True:
        raise audit.AuditFailure('public_save_unconfirmed')
    if http.api('POST', '/api/v4/v4/reports', token, data=data, files=files) != saved:
        raise audit.AuditFailure('public_retry_not_idempotent')
    check_report(http.api('GET', '/api/v4/v4/reports/' + state['version'], token), state)
    http.api('GET', '/api/v4/v4/reports/' + state['version'], second, 404)
    result = {'public_https_normal_tls': 'passed', 'health_v3_v4': 'passed',
              'anonymous_401_json': 'passed', 'wrong_password_401_json': 'passed',
              'demo_password_login': 'passed', 'owner_isolation': 'passed',
              'normal_user_admin_refused': 'passed', 'capabilities_versions': [1, 1],
              'saved_reference_versions_read': 2, 'moderated_contours_read': 3,
              'bounded_real_tree_analysis': 'passed', 'save_idempotency_readback': 'passed',
              'photo_mask_measurements_identical': True,
              'model_sha256': state['snapshot']['report']['segmentation_weights_sha256'],
              'real_user_records_changed': False, 'training_started': False,
              'phone_ui_checked': False}
    audit.write_private(root / 'public-check-result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.root)))
    except Exception as exc:
        print(json.dumps({'public_check': 'incomplete', 'category': str(exc)
                         if isinstance(exc, audit.AuditFailure) else type(exc).__name__}))
        sys.exit(1)
