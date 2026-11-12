import hashlib
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_image_delta as delta


def tar(path, values, mode='w'):
    with tarfile.open(path, mode) as target:
        for name, raw in values.items():
            item = tarfile.TarInfo(name); item.size = len(raw); target.addfile(item, io.BytesIO(raw))


@pytest.fixture
def archives(tmp_path):
    layer = b'exact uncompressed layer'
    config = json.dumps({'os': 'linux', 'architecture': 'amd64', 'rootfs': {
        'type': 'layers', 'diff_ids': ['sha256:' + hashlib.sha256(layer).hexdigest()]}}).encode()
    image = 'sha256:' + hashlib.sha256(config).hexdigest()
    base, patch, out = tmp_path / 'base.tar', tmp_path / 'delta.tar.gz', tmp_path / 'out.tar.partial'
    tar(base, {'layer.tar': layer, 'stale-base-member': b'must not survive'})
    values = {'config.json': config, 'manifest.json': json.dumps([{'Config': 'config.json', 'Layers': ['layer.tar']}]).encode()}
    rows = [{'name': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'base': False} for name, raw in values.items()]
    rows.append({'name': 'layer.tar', 'bytes': len(layer), 'sha256': hashlib.sha256(layer).hexdigest(), 'base': True})
    info = {'format': 1, 'kind': 'docker_save_verified_member_delta', 'image_id': image,
            'base_sha256': delta.digest(base), 'members': rows, 'changed_bytes': sum(map(len, values.values()))}
    def write(): tar(patch, {**values, 'DELTA_MANIFEST.json': json.dumps(info).encode()}, 'w:gz')
    write()
    return base, patch, out, image, info, values, write


def test_member_delta_preserves_exact_image_and_omits_stale_base_members(archives):
    base, patch, out, image, info, _, _ = archives
    result = delta.apply_member_delta(base, patch, out, image_id=image, base_sha256=info['base_sha256'])
    assert result['image_closure_verified'] is True and result['container_started'] is False
    with tarfile.open(out) as archive:
        assert set(archive.getnames()) == {'config.json', 'manifest.json', 'layer.tar'}


@pytest.mark.parametrize('case', ['base', 'payload', 'extra', 'budget', 'duplicate', 'identity'])
def test_invalid_member_delta_never_becomes_verified(archives, case):
    base, patch, out, image, info, values, write = archives
    if case == 'base': base.write_bytes(b'changed-base')
    if case == 'payload': values['config.json'] = b'changed-same-length'
    if case == 'extra': values['unrequested'] = b'extra'
    if case == 'budget': info['changed_bytes'] = 513 * 1024**2
    if case == 'duplicate': info['members'].append(info['members'][0])
    if case == 'identity': info['image_id'] = 'sha256:' + 'f' * 64
    write()
    with pytest.raises((ValueError, KeyError, tarfile.TarError)):
        delta.apply_member_delta(base, patch, out, image_id=image, base_sha256=info['base_sha256'])
    assert not out.with_name('COMPLETE').exists()


@pytest.mark.parametrize('kind', ['payload', 'pax', 'longname'])
def test_oversized_raw_header_is_refused_before_tarfile_scans_or_allocates(archives, monkeypatch, kind):
    base, patch, out, image, info, _, _ = archives
    item = tarfile.TarInfo('oversize')
    item.type = {'payload': tarfile.REGTYPE, 'pax': tarfile.XHDTYPE,
                 'longname': tarfile.GNUTYPE_LONGNAME}[kind]
    item.size = 16 * 1024**2
    # Only a header is needed; a bad declaration must fail before payload read.
    with gzip.open(patch, 'wb') as target:
        target.write(item.tobuf())
    def forbidden(*a, **k):
        raise AssertionError('unbounded getmembers must not run')
    monkeypatch.setattr(tarfile.TarFile, 'getmembers', forbidden)
    with pytest.raises(ValueError, match='budget|header'):
        delta.apply_member_delta(base, patch, out, image_id=image,
                                 base_sha256=info['base_sha256'], max_changed_bytes=1024)
    assert not out.exists()


def test_member_header_count_is_bounded_before_full_list(archives, monkeypatch):
    base, patch, out, image, info, _, _ = archives
    monkeypatch.setattr(delta, 'MAX_DELTA_MEMBERS', 3, raising=False)
    tar(patch, {f'file-{n}': b'' for n in range(4)}, 'w:gz')
    with pytest.raises(ValueError, match='member.*budget'):
        delta.apply_member_delta(base, patch, out, image_id=image,
                                 base_sha256=info['base_sha256'])
    assert not out.exists()


def test_pax_size_override_cannot_bypass_raw_payload_budget(archives):
    base, patch, out, image, info, _, _ = archives
    item = tarfile.TarInfo('file'); item.size = 1
    item.pax_headers = {'size': str(64 * 1024**3)}
    with tarfile.open(patch, 'w:gz') as target:
        target.addfile(item, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='PAX|header'):
        delta.apply_member_delta(base, patch, out, image_id=image,
                                 base_sha256=info['base_sha256'], max_changed_bytes=1024)
    assert not out.exists()


def test_trailing_inflate_bomb_is_bounded_and_never_reaches_output(archives, monkeypatch):
    base, patch, out, image, info, _, _ = archives
    monkeypatch.setattr(delta, 'MAX_DELTA_MEMBERS', 2, raising=False)
    monkeypatch.setattr(delta, 'MAX_DELTA_MANIFEST', 1024, raising=False)
    monkeypatch.setattr(delta, 'MAX_EXTENDED_BYTES', 1024, raising=False)
    with gzip.open(patch, 'wb') as target:
        for _ in range(128):
            target.write(bytes(64 * 1024))
    with pytest.raises(ValueError, match='inflate.*budget'):
        delta.apply_member_delta(base, patch, out, image_id=image,
                                 base_sha256=info['base_sha256'], max_changed_bytes=1024)
    assert not out.exists()


def test_target_manifest_list_budget_is_checked_before_output(archives, monkeypatch):
    base, patch, out, image, info, _, write = archives
    monkeypatch.setattr(delta, 'MAX_DELTA_MEMBERS', 3)
    info['members'].append({'name': 'stale-base-member', 'bytes': 16,
                            'sha256': 'a' * 64, 'base': True})
    write()
    with pytest.raises(ValueError, match='metadata'):
        delta.apply_member_delta(base, patch, out, image_id=image,
                                 base_sha256=info['base_sha256'])
    assert not out.exists()


def test_small_standard_pax_metadata_remains_compatible(archives):
    base, patch, out, image, info, values, _ = archives
    with tarfile.open(patch, 'w:gz') as target:
        for name, raw in {**values, 'DELTA_MANIFEST.json': json.dumps(info).encode()}.items():
            member = tarfile.TarInfo(name); member.size = len(raw)
            member.pax_headers = {'mtime': '1.5'}
            target.addfile(member, io.BytesIO(raw))
    result = delta.apply_member_delta(base, patch, out, image_id=image,
                                      base_sha256=info['base_sha256'])
    assert result['image_closure_verified'] is True
