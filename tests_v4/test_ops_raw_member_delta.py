"""Exact Docker 29.8 framing with real headers and tiny synthetic OCI bytes."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_image_delta as delta

NATIVE = json.loads((Path(__file__).parent / 'fixtures/docker_save_native_headers.json')
                    .read_text(encoding='utf-8-sig'))['headers']
EMPTY_SHA = hashlib.sha256(b'').hexdigest()


def native_header(name, size=0, kind='file'):
    template = NATIVE['blobs' if kind == 'directory' else 'oci-layout']
    raw = bytearray(base64.b64decode(template))
    raw[:100] = (name + ('/' if kind == 'directory' else '')).encode().ljust(100, b'\0')
    raw[124:136] = ('%011o\0' % size).encode()
    raw[148:156] = b' ' * 8
    raw[148:156] = ('%06o\0 ' % sum(raw)).encode()
    return bytes(raw)


def write_tar(path, values, mode='w'):
    with tarfile.open(path, mode) as archive:
        for name, payload in values.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def make_fixture(tmp_path, layer=b'synthetic layer bytes', os_name='linux', architecture='amd64'):
    sha = lambda value: hashlib.sha256(value).hexdigest()
    blob = lambda value: 'blobs/sha256/' + sha(value)
    encoded = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    config = encoded({'os': os_name, 'architecture': architecture, 'rootfs': {
        'type': 'layers', 'diff_ids': ['sha256:' + sha(layer)]}})
    manifest = encoded({'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.manifest.v1+json',
        'config': {'digest': 'sha256:' + sha(config), 'size': len(config)},
        'layers': [{'digest': 'sha256:' + sha(layer), 'size': len(layer)}]})
    index = encoded({'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.index.v1+json',
        'manifests': [{'digest': 'sha256:' + sha(manifest), 'size': len(manifest),
                       'platform': {'os': os_name, 'architecture': architecture}}]})
    image = 'sha256:' + sha(index)
    values = {'oci-layout': b'{"imageLayoutVersion":"1.0.0"}', blob(layer): layer,
              blob(config): config, blob(manifest): manifest, blob(index): index,
              'index.json': index,
              'manifest.json': encoded([{'Config': blob(config), 'Layers': [blob(layer)]}])}
    base, patch, output = [tmp_path / name for name in ('base.tar', 'delta.tar.gz', 'output.tar.partial')]
    write_tar(base, {blob(layer): layer, 'stale-base-member': b'not part of target'})
    rows, chunks = [], []
    for name in ('blobs', 'blobs/sha256'):
        header = base64.b64decode(NATIVE[name])
        rows.append({'name': name, 'bytes': 0, 'sha256': EMPTY_SHA, 'base': False,
                     'type': 'directory', 'header_b64': base64.b64encode(header).decode()})
        chunks.append(header)
    changed = {}
    for name, payload in values.items():
        reused = name == blob(layer)
        header = native_header(name, len(payload))
        rows.append({'name': name, 'bytes': len(payload), 'sha256': sha(payload), 'base': reused,
                     'type': 'file', 'header_b64': base64.b64encode(header).decode()})
        chunks.extend([header, payload, bytes(-len(payload) % 512)])
        if not reused:
            changed[name] = payload
    target = b''.join(chunks) + bytes(1024)
    info = {'format': 2, 'kind': 'docker_save_verified_member_delta', 'image_id': image,
            'base_sha256': delta.digest(base), 'members': rows,
            'changed_bytes': sum(map(len, changed.values())), 'target_bytes': len(target),
            'trailing_zero_bytes': 1024}

    def write():
        write_tar(patch, {**changed, 'DELTA_MANIFEST.json': encoded(info)}, 'w:gz')

    def apply(**kwargs):
        return delta.apply_raw_member_delta(base, patch, output, image_id=image,
            base_sha256=info['base_sha256'], expected_archive_sha256=sha(target), **kwargs)

    write()
    return SimpleNamespace(base=base, patch=patch, output=output, info=info, changed=changed,
                           target=target, image=image, write=write, apply=apply)


@pytest.fixture
def raw_fixture(tmp_path):
    return make_fixture(tmp_path)


def test_native_two_directory_archive_roundtrips_exact_headers_order_and_1024_eof(raw_fixture):
    f = raw_fixture
    assert f.target[:512] == base64.b64decode(NATIVE['blobs'])
    assert f.target[512:1024] == base64.b64decode(NATIVE['blobs/sha256'])
    assert len(f.target) % tarfile.RECORDSIZE != 0
    result = f.apply()
    assert f.output.read_bytes() == f.target
    assert result['image_closure_verified'] is True
    assert result['container_started'] is False
    with tarfile.open(f.output) as restored:
        assert restored.getnames() == [row['name'] for row in f.info['members']]
        assert restored.getmember('blobs').isdir()
        assert restored.getmember('blobs/sha256').isdir()
        assert 'stale-base-member' not in restored.getnames()


def test_v1_default_entrypoint_refuses_v2_before_target_creation(raw_fixture):
    f = raw_fixture
    with pytest.raises(ValueError, match='metadata|format'):
        delta.apply_member_delta(f.base, f.patch, f.output, image_id=f.image,
            base_sha256=f.info['base_sha256'], expected_archive_sha256=hashlib.sha256(f.target).hexdigest())
    assert not f.output.exists()


@pytest.mark.parametrize('size', [1, 1024**3])
def test_directory_size_never_allocates_payload_or_creates_output(raw_fixture, size):
    f = raw_fixture
    row = f.info['members'][0]
    row['bytes'] = size
    row['header_b64'] = base64.b64encode(native_header(row['name'], size, 'directory')).decode()
    f.write()
    with pytest.raises(ValueError, match='directory|Directory'):
        f.apply()
    assert not f.output.exists()


@pytest.mark.parametrize('case', ['directory-payload', 'directory-from-base', 'duplicate',
                                 'file-directory-same-name', 'file-parent-collision'])
def test_directory_and_file_ambiguities_are_refused_before_output(raw_fixture, case):
    f = raw_fixture
    if case == 'directory-payload':
        f.changed['blobs'] = b'forbidden payload'
    elif case == 'directory-from-base':
        f.info['members'][0]['base'] = True
    elif case == 'duplicate':
        f.info['members'].append(copy.deepcopy(f.info['members'][0]))
    else:
        row = {'name': 'blobs', 'bytes': 0, 'sha256': EMPTY_SHA, 'base': False,
               'type': 'file', 'header_b64': base64.b64encode(native_header('blobs')).decode()}
        f.changed['blobs'] = b''
        if case == 'file-directory-same-name':
            f.info['members'].append(row)
        else:
            f.info['members'][0] = row
    f.write()
    with pytest.raises(ValueError, match='directory|Directory|Duplicate|collision|entries'):
        f.apply()
    assert not f.output.exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:drive', 'bad\\path'])
def test_raw_directory_paths_remain_strict(raw_fixture, name):
    f = raw_fixture
    f.info['members'][0].update(name=name,
        header_b64=base64.b64encode(native_header(name, kind='directory')).decode())
    f.write()
    with pytest.raises(ValueError, match='path|header'):
        f.apply()
    assert not f.output.exists()


@pytest.mark.parametrize('link_type', [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_raw_headers_never_accept_symlinks_or_hardlinks(raw_fixture, link_type):
    f = raw_fixture
    row = f.info['members'][2]
    header = bytearray(base64.b64decode(row['header_b64']))
    header[156:157] = link_type
    header[157:257] = b'outside'.ljust(100, b'\0')
    header[148:156] = b' ' * 8
    header[148:156] = ('%06o\0 ' % sum(header)).encode()
    row['header_b64'] = base64.b64encode(header).decode()
    f.write()
    with pytest.raises(ValueError, match='type|link|header'):
        f.apply()
    assert not f.output.exists()


@pytest.mark.parametrize('unsupported', [tarfile.GNUTYPE_SPARSE, tarfile.CONTTYPE])
def test_regular_like_sparse_and_contiguous_types_refuse_before_output(raw_fixture, unsupported):
    f = raw_fixture
    row = f.info['members'][2]
    header = bytearray(base64.b64decode(row['header_b64']))
    header[156:157] = unsupported
    header[148:156] = b' ' * 8
    header[148:156] = ('%06o\0 ' % sum(header)).encode()
    row['header_b64'] = base64.b64encode(header).decode()
    f.write()
    with pytest.raises(ValueError, match='type|header'):
        f.apply()
    assert not f.output.exists()


@pytest.mark.parametrize('case', ['checksum', 'short-header', 'long-header', 'name-mismatch',
                                 'size-mismatch', 'bad-base64', 'target-size', 'short-eof', 'large-eof'])
def test_invalid_native_framing_cannot_create_output(raw_fixture, case):
    f = raw_fixture
    row = f.info['members'][2]
    header = base64.b64decode(row['header_b64'])
    if case == 'checksum':
        row['header_b64'] = base64.b64encode(bytes([header[0] ^ 1]) + header[1:]).decode()
    elif case == 'short-header':
        row['header_b64'] = base64.b64encode(header[:-1]).decode()
    elif case == 'long-header':
        row['header_b64'] = base64.b64encode(header + b'\0').decode()
    elif case == 'name-mismatch':
        row['header_b64'] = base64.b64encode(native_header('different', row['bytes'])).decode()
    elif case == 'size-mismatch':
        row['header_b64'] = base64.b64encode(native_header(row['name'], row['bytes'] + 1)).decode()
    elif case == 'bad-base64':
        row['header_b64'] = '$' * 684
    elif case == 'target-size':
        f.info['target_bytes'] += 512
    elif case == 'short-eof':
        f.info['trailing_zero_bytes'] = 512
    else:
        f.info['trailing_zero_bytes'] = 1024**3
    f.write()
    with pytest.raises(ValueError, match='header|framing|size|EOF|metadata'):
        f.apply()
    assert not f.output.exists()


def test_unknown_protocol_version_refuses_before_output(raw_fixture):
    f = raw_fixture
    f.info['format'] = 999
    f.write()
    with pytest.raises(ValueError, match='metadata|format'):
        f.apply()
    assert not f.output.exists()


def test_raw_protocol_version_must_be_an_integer(raw_fixture):
    f = raw_fixture
    f.info['format'] = 2.0
    f.write()
    with pytest.raises(ValueError, match='metadata'):
        f.apply()
    assert not f.output.exists()


def test_raw_format_requires_the_trusted_expected_full_archive_hash(raw_fixture):
    f = raw_fixture
    with pytest.raises(ValueError, match='checksum|required'):
        delta.apply_raw_member_delta(f.base, f.patch, f.output, image_id=f.image,
                                 base_sha256=f.info['base_sha256'])
    assert not f.output.exists()


def test_raw_output_reserves_disk_before_opening_large_target(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, layer=b'x' * (20 * 1024**2))
    monkeypatch.setattr(delta.shutil, 'disk_usage', lambda _: SimpleNamespace(
        free=delta.RESERVED_FREE_BYTES + 17 * 1024**2))
    with pytest.raises(ValueError, match='output space'):
        f.apply(max_changed_bytes=1024**2)
    assert not f.output.exists()


def test_raw_writer_rechecks_retained_disk_floor_for_each_write(raw_fixture, monkeypatch):
    f = raw_fixture
    opened = False
    real_open = Path.open

    def observed_open(path, *args, **kwargs):
        nonlocal opened
        stream = real_open(path, *args, **kwargs)
        if path == f.output and args and args[0] == 'xb':
            opened = True
        return stream

    monkeypatch.setattr(Path, 'open', observed_open)
    monkeypatch.setattr(delta.shutil, 'disk_usage', lambda _: SimpleNamespace(
        free=delta.RESERVED_FREE_BYTES if opened else 100 * 1024**3))
    with pytest.raises(ValueError, match='output space'):
        f.apply()
    assert f.output.is_file() and f.output.stat().st_size == 0


def test_full_target_hash_failure_cannot_be_reported_as_verified(raw_fixture):
    f = raw_fixture
    with pytest.raises(ValueError, match='Reconstructed archive checksum'):
        delta.apply_raw_member_delta(f.base, f.patch, f.output, image_id=f.image,
            base_sha256=f.info['base_sha256'], expected_archive_sha256='f' * 64)


def test_requested_oci_identity_is_checked_even_with_a_matching_archive_hash(raw_fixture):
    from ops_selfhost_backup import BackupError
    f = raw_fixture
    f.info['image_id'] = 'sha256:' + 'f' * 64
    f.write()
    with pytest.raises(BackupError, match='image_id_mismatch'):
        delta.apply_raw_member_delta(f.base, f.patch, f.output, image_id=f.info['image_id'],
            base_sha256=f.info['base_sha256'], expected_archive_sha256=hashlib.sha256(f.target).hexdigest())


@pytest.mark.parametrize('os_name,architecture', [('darwin', 'amd64'), ('linux', 'arm64')])
def test_native_image_gate_still_requires_linux_amd64(tmp_path, os_name, architecture):
    from ops_selfhost_backup import BackupError
    f = make_fixture(tmp_path, os_name=os_name, architecture=architecture)
    with pytest.raises(BackupError, match='platform_missing'):
        f.apply()


def test_layer_digest_is_checked_independently_of_member_and_full_archive_hashes(tmp_path):
    from ops_selfhost_backup import BackupError
    payload = b'unique synthetic layer data'
    f = make_fixture(tmp_path, layer=payload)
    row = next(row for row in f.info['members'] if row['base'])
    changed = b'x' * len(payload)
    write_tar(f.base, {row['name']: changed, 'stale-base-member': b'irrelevant'})
    f.info['base_sha256'] = delta.digest(f.base)
    row['sha256'] = hashlib.sha256(changed).hexdigest()
    f.write()
    target_hash = hashlib.sha256(f.target.replace(payload, changed, 1)).hexdigest()
    with pytest.raises(BackupError, match='layer_checksum_mismatch'):
        delta.apply_raw_member_delta(f.base, f.patch, f.output, image_id=f.image,
            base_sha256=f.info['base_sha256'], expected_archive_sha256=target_hash)


def test_cli_raw_format_is_explicit_and_default_cli_cannot_create_its_target(raw_fixture):
    f = raw_fixture
    script = Path(__file__).parents[1] / 'deploy-vps/ops_image_delta.py'
    common = [str(f.base), str(f.patch), str(f.output), '--image-id', f.image,
              '--base-sha256', f.info['base_sha256'], '--expected-archive-sha256',
              hashlib.sha256(f.target).hexdigest()]
    rejected = subprocess.run([sys.executable, '-B', str(script), 'apply-member', *common], capture_output=True)
    assert rejected.returncode != 0 and not f.output.exists()
    accepted = subprocess.run([sys.executable, '-B', str(script), 'apply-raw-member', *common],
                              capture_output=True, text=True, check=True)
    assert json.loads(accepted.stdout)['delta_format'] == 2
    assert f.output.read_bytes() == f.target
