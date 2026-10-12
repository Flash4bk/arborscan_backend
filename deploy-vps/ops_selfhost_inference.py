"""Bounded real inference and durable save using only new DEMO account writes.

Reads a restored photograph privately as the input fixture. Does not impersonate
its original owner or change an existing report. No training and no activation.
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import uuid

import ops_selfhost_audit as audit
from PIL import Image, ImageOps


def run(root, repeat=False, photo=None):
    root = Path(root)
    config = audit.private_json(root / 'compose.private.json')
    base = audit.guarded_stack(root, config, root / 'assets')
    http = audit.StageHTTP(base, config['services']['api']['environment']['SUPABASE_SERVICE_KEY'],
                           api_bases=audit.api_bases(config))
    fixture = audit.validate_fixture(audit.private_json(root / 'fixture.private.json'))
    if not fixture.get('complete'):
        raise audit.AuditFailure('complete_demo_fixture_required')
    account = fixture['accounts'][0]
    login = http.api('POST', '/api/v3/auth/login', json={k: account[k] for k in ('email', 'password')})
    if login['user']['id'] != account['id']:
        raise audit.AuditFailure('inference_fixture_owner_mismatch')
    token = login['token']
    path = root / 'inference.fixture.private.json'
    if path.exists():
        state = audit.private_json(path)
        if state.get('owner') != account['id'] or state.get('tag') != audit.FIXTURE_NAME:
            raise audit.AuditFailure('inference_fixture_binding_mismatch')
    else:
        if repeat:
            raise audit.AuditFailure('existing_inference_fixture_required')
        sample = None
        if photo is not None:
            photo = Path(photo)
            if photo.is_symlink() or not photo.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
                raise audit.AuditFailure('private_photo_fixture_required')
            sample = photo.read_bytes()
        else:
            for candidate in (root / 'storage-seed' / 'objects').rglob('*.json'):
                if 'report-versions' not in candidate.parts:
                    continue
                payload = json.loads(candidate.read_text())
                image = payload.get('image', {})
                if image.get('width', 0) * image.get('height', 0) >= 100000:
                    sample = base64.b64decode(image['original_base64'], validate=True)
                    break
        if sample is None:
            raise audit.AuditFailure('bounded_real_photo_fixture_unavailable')
        with Image.open(io.BytesIO(sample)) as im:
            oriented = ImageOps.exif_transpose(im).convert('RGB')
            oriented.thumbnail((640, 640))
            output = io.BytesIO(); oriented.save(output, 'JPEG', quality=90)
            raw = output.getvalue()
        result = http.api('POST', '/api/v4/v4/analyze-tree', token,
                          files={'file': ('DEMO-inference.jpg', raw, 'image/jpeg')},
                          data={'tap_x': '.5', 'tap_y': '.4'})
        if result.get('tree', {}).get('detected') is not True:
            raise audit.AuditFailure('real_tree_inference_not_detected')
        if not result.get('analysis_id') or not result.get('segmentation_weights_sha256'):
            raise audit.AuditFailure('inference_model_provenance_unavailable')
        state = {'tag': audit.FIXTURE_NAME, 'owner': account['id'], 'analysis': result['analysis_id'],
                 'version': str(uuid.uuid4()), 'image': base64.b64encode(raw).decode(),
                 'snapshot': {'version': 1, 'kind': 'v4', 'change_source': 'analysis',
                    'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'report': result}}
        audit.write_private(path, state)
    raw = base64.b64decode(state['image'], validate=True)
    if not repeat:
        saved = http.api('POST', '/api/v4/v4/reports', token,
                data={'analysis_id': state['analysis'], 'version_id': state['version'],
                      'snapshot': json.dumps(state['snapshot'])},
                files={'image': ('DEMO-inference.jpg', raw, 'image/jpeg')})
        if saved.get('saved') is not True or saved.get('persisted') is not True:
            raise audit.AuditFailure('inference_save_unconfirmed')
    restored = http.api('GET', '/api/v4/v4/reports/' + state['version'], token)
    snapshot = restored['snapshot']
    if (restored.get('persisted') is not True
            or base64.b64decode(snapshot['image']['original_base64'], validate=True) != raw
            or snapshot['report']['measurements'] != state['snapshot']['report']['measurements']
            or snapshot['report']['images'] != state['snapshot']['report']['images']):
        raise audit.AuditFailure('inference_report_roundtrip_mismatch')
    receipt = {'bounded_real_tree_inference': 'passed', 'analysis_saved': True,
        'photo_mask_measurements_roundtrip': 'passed', 'restart_read': repeat,
        'model_sha256': state['snapshot']['report']['segmentation_weights_sha256'],
        'new_demo_owner_only': True, 'real_user_records_changed': False, 'training_started': False}
    audit.write_private(root / ('inference-restart.json' if repeat else 'inference-result.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--photo', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.root, args.repeat, args.photo)))
    except Exception as exc:
        print(json.dumps({'inference': 'failed', 'category': str(exc) if isinstance(exc, audit.AuditFailure)
                         else type(exc).__name__}))
        sys.exit(1)
