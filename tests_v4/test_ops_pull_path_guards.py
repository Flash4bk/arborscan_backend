"""Real resume files and redirects; never touch an installed backup root."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_pull_backups as pull

NAME = '20261012T210000Z'


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'offsite'
    root.mkdir()
    files = {
        'application.tar': b'synthetic application archive bytes',
        'local-files.tar': b'synthetic local files',
        'arborscan.env.private': b'synthetic private config',
        'containers.private.json': b'[]',
        'assets/photo.bin': b'synthetic photo bytes',
    }
    record = {'name': NAME, 'manifest': ''.join(
        hashlib.sha256(raw).hexdigest() + '  ' + name + '\n' for name, raw in files.items()),
        'bytes': sum(map(len, files.values()))}
    return root, files, record


def link(path, target, kind):
    if kind == 'junction':
        if os.name != 'nt':
            pytest.skip('Windows directory junction fixture')
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(path), str(target)],
                                capture_output=True)
        assert result.returncode == 0, 'Could not create isolated test junction'
    else:
        try:
            path.symlink_to(target, target_is_directory=target.is_dir())
        except OSError:
            pytest.skip('Symlink creation unavailable in this environment')


def refused_before_download(source):
    root, _, record = source
    calls = []
    with pytest.raises(ValueError, match='symlink_or_reparse_refused'):
        pull.transfer(root, record, lambda *args: calls.append(args),
                      download_many=lambda *args: calls.append(args),
                      free=lambda _: SimpleNamespace(free=10 * 1024**3))
    assert calls == []


@pytest.mark.parametrize('kind', ['symlink', 'junction'])
@pytest.mark.parametrize('destination', ['outside', 'protected'])
def test_partial_directory_redirect_refused_before_download(source, tmp_path, kind, destination):
    root, _, _ = source
    target = tmp_path / 'outside' if destination == 'outside' else root / '20261001T010000Z'
    target.mkdir()
    sentinel = target / 'application.tar'
    sentinel.write_bytes(b'protected original bytes')
    link(root / (NAME + '.partial'), target, kind)
    refused_before_download(source)
    assert sentinel.read_bytes() == b'protected original bytes'
    assert sorted(p.name for p in target.iterdir()) == ['application.tar']


@pytest.mark.parametrize('kind', ['symlink', 'junction'])
def test_nested_partial_redirect_refused_before_download(source, tmp_path, kind):
    root, _, _ = source
    partial = root / (NAME + '.partial')
    partial.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    sentinel = outside / 'photo.bin'
    sentinel.write_bytes(b'original independent file')
    link(partial / 'assets', outside, kind)
    refused_before_download(source)
    assert sentinel.read_bytes() == b'original independent file'
    assert sorted(p.name for p in outside.iterdir()) == ['photo.bin']


@pytest.mark.parametrize('filename', ['application.tar.downloading', 'SHA256SUMS',
                                      'OFFSITE_VERIFIED.json', 'COMPLETE'])
def test_file_redirects_refused_before_download(source, tmp_path, filename):
    root, _, _ = source
    partial = root / (NAME + '.partial')
    partial.mkdir()
    sentinel = tmp_path / 'outside.bin'
    sentinel.write_bytes(b'untouched outside bytes')
    link(partial / filename, sentinel, 'symlink')
    refused_before_download(source)
    assert sentinel.read_bytes() == b'untouched outside bytes'


def test_legitimate_partial_reget_and_verified_existing_legacy_files(source):
    root, files, record = source
    partial = root / (NAME + '.partial')
    partial.mkdir()
    (partial / 'local-files.tar').write_bytes(files['local-files.tar'])
    downloading = partial / 'application.tar.downloading'
    downloading.write_bytes(files['application.tar'][:9])
    calls = []
    def many(name, pending):
        assert name == NAME
        for filename, target in pending:
            calls.append(filename)
            offset = target.stat().st_size if target.exists() else 0
            assert target.name.endswith('.downloading')
            with target.open('ab') as stream:
                stream.write(files[filename][offset:])
    assert pull.transfer(root, record, lambda *_: pytest.fail('single download not expected'),
        download_many=many, free=lambda _: SimpleNamespace(free=10 * 1024**3)) == 'downloaded_verified'
    final = root / NAME
    assert 'local-files.tar' not in calls and 'application.tar' in calls
    assert all((final / name).read_bytes() == raw for name, raw in files.items())
    assert (final / 'COMPLETE').is_file() and not partial.exists()
    assert pull.transfer(root, record, lambda *_: pytest.fail('verified bytes downloaded again'),
        download_many=lambda *_: pytest.fail('verified bytes downloaded again')) == 'verified_existing'
