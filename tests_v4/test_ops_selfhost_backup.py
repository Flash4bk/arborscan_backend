"""Synthetic local-file/orchestration checks, not a real PostgreSQL restore.

Actual pg_dump/pg_restore and self-host service integration remain separate VPS
checks. These tests never connect to Supabase, Docker or a production account.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_selfhost_backup as ops

PRIVATE = 'synthetic-private-not-for-output'
IMAGE = 'sha256:' + 'a' * 64


def file_tar(path, members, *, filesystem=False):
    if filesystem:
        parents = {'.'} if members else set()
        for name in members:
            current = PurePosixPath(name).parent
            while current != PurePosixPath('.'):
                parents.add(current.as_posix()); current = current.parent
        members = {**members, 'filesystem-metadata.json': ops.canonical({'format': 1,
            'files': {name: {'uid': 0, 'gid': 0, 'mode': 0o600, 'mtime_ns': 1,
                            'xattrs': {}} for name in members},
            'directories': {name: {'uid': 0, 'gid': 0, 'mode': 0o700,
                                  'mtime_ns': 1, 'xattrs': {}} for name in parents}})}
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()}
    with tarfile.open(path, 'w') as archive:
        for name, raw in {**members, 'manifest.json': ops.canonical(hashes)}.items():
            item = tarfile.TarInfo(name); item.size = len(raw); item.mode = 0o600
            archive.addfile(item, io.BytesIO(raw))


def inventory(**changes):
    return ops.normalized_inventory({'buckets': [{'id': 'private'}], 'objects': [{
        'id': 'object-a', 'bucket_id': 'private', 'name': 'photo.bin',
        'version': 'generation-a', 'metadata': {'size': 3}, **changes,
    }]})


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../b', 'a\\b', 'a:drive', 'a/./b', '\x00', 'a//b', 'a/'])
def test_unsafe_members_refused(name):
    with pytest.raises(ops.BackupError):
        ops.member_name(name)


def test_generation_mapping_and_duplicate_names_are_guarded():
    items = ops.physical_objects(inventory(), 'stub', 'version-directory', 'global')
    assert items == [{'physical': 'global/stub/private/photo.bin/generation-a',
                      'logical': 'private/photo.bin', 'size': 3}]
    assert ops.physical_objects(inventory(), 'stub', 'version-suffix', 'global')[0]['physical'].endswith('photo.bin__generation-a')
    assert ops.physical_objects(inventory(version=None), 'arborscan', 'version-directory', 'arborscan')[0]['physical'] == 'arborscan/arborscan/private/photo.bin'
    with pytest.raises(ops.BackupError):
        ops.physical_objects(inventory(version='../escape'), 'stub', 'version-directory', 'global')
    data = inventory(); data['objects'].append(data['objects'][0])
    with pytest.raises(ops.BackupError):
        ops.physical_objects(data, 'stub', 'version-directory', 'global')


def test_real_file_helper_hashes_bytes_and_legacy_application_format(tmp_path):
    source = tmp_path / 'volume'; source.mkdir()
    file = source / 'global/stub/private/photo.bin/generation-a'; file.parent.mkdir(parents=True)
    file.write_bytes(b'abc')
    helper = ops.FILE_HELPER.replace("pathlib.Path('/source')", 'pathlib.Path(' + repr(str(source)) + ')')
    objects = ops.physical_objects(inventory(), 'stub', 'version-directory', 'global')
    result = subprocess.run([sys.executable, '-c', helper], input=ops.canonical({'objects': objects}), capture_output=True)
    assert result.returncode == 0
    archive = tmp_path / 'storage-files.tar'; archive.write_bytes(result.stdout)
    hashes = ops.verify_file_tar(archive)
    assert hashes[objects[0]['physical']] == hashlib.sha256(b'abc').hexdigest()
    metadata = ops.verify_filesystem_metadata(archive, hashes)
    assert metadata['files'][objects[0]['physical']]['mode'] == (file.stat().st_mode & 0o777)
    assert '.' in metadata['directories']
    raw_target = tmp_path / 'raw-restored'
    restored_volume = ops.restore_volume_archive(archive, raw_target, apply_ownership=False)
    assert restored_volume['files_restored'] == 1
    assert (raw_target / objects[0]['physical']).read_bytes() == b'abc'
    assert restored_volume['ownership_restored'] is False
    ops.application_archive(tmp_path, objects, {'report_versions': [], 'users': [{'id': PRIVATE}]}, inventory())
    restored = tmp_path / 'restored'
    stats = ops.restore(tmp_path / 'application.tar', restored)
    assert (restored / 'objects/private/photo.bin').read_bytes() == b'abc'
    assert json.loads((restored / 'tables/users.json').read_bytes()) == [{'id': PRIVATE}]
    assert stats['database_restored'] is False


def test_real_file_helper_rejects_wrong_object_size(tmp_path):
    (tmp_path / 'photo').write_bytes(b'wrong-size')
    helper = ops.FILE_HELPER.replace("pathlib.Path('/source')", 'pathlib.Path(' + repr(str(tmp_path)) + ')')
    result = subprocess.run([sys.executable, '-c', helper], input=ops.canonical({
        'objects': [{'physical': 'photo', 'size': 1}]}), capture_output=True)
    assert result.returncode != 0
    assert result.stdout == b''


def test_file_helper_guards_work_with_python_optimization(tmp_path):
    (tmp_path / 'photo').write_bytes(b'wrong-size')
    helper = ops.FILE_HELPER.replace("pathlib.Path('/source')", 'pathlib.Path(' + repr(str(tmp_path)) + ')')
    result = subprocess.run([sys.executable, '-O', '-c', helper], input=ops.canonical({
        'objects': [{'physical': 'photo', 'size': 1}]}), capture_output=True)
    assert result.returncode != 0
    assert result.stdout == b''


@pytest.mark.skipif(not hasattr(os, 'setxattr'), reason='Real xattrs require Linux filesystem')
def test_raw_volume_metadata_preserves_real_xattrs(tmp_path):
    file = tmp_path / 'photo'; file.write_bytes(b'abc'); file.chmod(0o640)
    os.setxattr(file, 'user.content-type', b'image/png')
    os.setxattr(file, 'user.cache-control', b'max-age=3600')
    helper = ops.FILE_HELPER.replace("pathlib.Path('/source')", 'pathlib.Path(' + repr(str(tmp_path)) + ')')
    result = subprocess.run([sys.executable, '-c', helper], input=ops.canonical({
        'objects': [{'physical': 'photo', 'size': 3}]}), capture_output=True)
    assert result.returncode == 0
    archive = tmp_path.parent / (tmp_path.name + '-result.tar'); archive.write_bytes(result.stdout)
    meta = ops.verify_filesystem_metadata(archive, ops.verify_file_tar(archive))
    assert meta['files']['photo']['mode'] == 0o640
    assert meta['files']['photo']['xattrs']['user.content-type'] == 'aW1hZ2UvcG5n'
    with tarfile.open(archive) as stream:
        assert stream.getmember('photo').pax_headers['SCHILY.xattr.user.cache-control'] == 'max-age=3600'
    target = tmp_path.parent / (tmp_path.name + '-restored')
    ops.restore_volume_archive(archive, target)
    assert (target / 'photo').read_bytes() == b'abc'
    assert os.getxattr(target / 'photo', 'user.content-type') == b'image/png'
    assert (target / 'photo').stat().st_mode & 0o777 == 0o640
    assert (target / 'photo').stat().st_uid == file.stat().st_uid


def test_volume_restore_refuses_existing_data_and_corruption(tmp_path):
    file_tar(tmp_path / 'valid.tar', {'photo': b'abc'}, filesystem=True)
    target = tmp_path / 'existing'; target.mkdir(); (target / 'preserve').write_bytes(b'private-existing')
    with pytest.raises(ops.BackupError, match='target_must_be_empty'):
        ops.restore_volume_archive(tmp_path / 'valid.tar', target, apply_ownership=False)
    assert (target / 'preserve').read_bytes() == b'private-existing'
    (tmp_path / 'corrupt.tar').write_bytes((tmp_path / 'valid.tar').read_bytes().replace(b'abc', b'bad', 1))
    empty = tmp_path / 'not-created'
    with pytest.raises(ops.BackupError, match='checksum'):
        ops.restore_volume_archive(tmp_path / 'corrupt.tar', empty, apply_ownership=False)
    assert not empty.exists()


def test_filesystem_metadata_must_cover_every_file_and_have_valid_xattrs(tmp_path):
    for body in ({'format': 1, 'files': {}, 'directories': {}},
                 {'format': 1, 'files': {'photo': {'uid': 0, 'gid': 0, 'mode': 0o600,
                     'mtime_ns': 1, 'xattrs': {'user.test': 'not-base64!'}}}, 'directories': {}}):
        file_tar(tmp_path / 'data.tar', {'photo': b'abc', 'filesystem-metadata.json': ops.canonical(body)})
        with pytest.raises(ops.BackupError):
            ops.verify_filesystem_metadata(tmp_path / 'data.tar', ops.verify_file_tar(tmp_path / 'data.tar'))


def test_corrupted_file_tar_never_passes_sha_check(tmp_path):
    file_tar(tmp_path / 'good.tar', {'photo': b'abc'})
    raw = (tmp_path / 'good.tar').read_bytes().replace(b'abc', b'bad', 1)
    (tmp_path / 'bad.tar').write_bytes(raw)
    with pytest.raises(ops.BackupError, match='checksum'):
        ops.verify_file_tar(tmp_path / 'bad.tar')


@pytest.fixture
def environment(tmp_path, monkeypatch):
    private = tmp_path / 'private'; private.mkdir()
    root = private / 'selfhost-20261012T160000Z'; root.mkdir()
    (root / 'compose.private.json').write_text('{}')
    output = private / 'ops-backups'; output.mkdir()
    for day in range(1, 15):
        old = output / ('202609' + str(day).zfill(2) + 'T030000Z'); old.mkdir()
        (old / 'COMPLETE').write_text('prior-verified')
        (old / 'preserved.private').write_text(PRIVATE + str(day))
    old_bytes = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    asset = private / 'runtime.archive'; asset.write_bytes(b'verified-synthetic-image')
    index = private / 'runtime-index.json'
    index.write_bytes(ops.canonical({'format': 1, 'images': {IMAGE: [{
        'path': str(asset), 'sha256': ops.digest(asset),
    }]}}))
    monkeypatch.setattr(ops, 'PRIVATE_HOME', private)
    monkeypatch.setattr(ops, 'BACKUP_ROOT', output)
    return SimpleNamespace(root=root, output=output, index=index, asset=asset,
                           old_bytes=old_bytes)


class FakeBackend:
    changed = False
    native_failure = False
    events = []
    def __init__(self, root, stage):
        self.root, self.stage = root, stage
        self.env = {'TENANT_ID': 'stub', 'GLOBAL_S3_BUCKET': 'global'}
        self.volume, self.runtime_volume = 'synthetic_storage', 'synthetic_runtime'
        self.rows = [{'Name': '/selfhost-' + service, 'Image': IMAGE, 'Config': {'Labels': {
            'com.docker.compose.service': service, 'com.docker.compose.project': 'selfhost-fixture'}}}
            for service in sorted(ops.SERVICES)]

    @contextlib.contextmanager
    def snapshot(self):
        self.events.append('snapshot-open')
        snap = SimpleNamespace(identifier='00000003-0000001B-1', storage=inventory,
            tables=lambda: {'report_versions': [], 'auth_sessions': [{'token': PRIVATE}]})
        try:
            yield snap
        finally:
            self.events.append('snapshot-close')

    def native(self, identifier, folder):
        self.events.append(('native', identifier))
        if self.native_failure:
            raise ops.BackupError(PRIVATE)
        folder.mkdir()
        for name in ops.NATIVE:
            (folder / name).write_bytes(b'PGDMPsynthetic' if name == 'database.dump' else b'synthetic')
        (folder / 'SHA256SUMS').write_text(''.join(ops.digest(folder / name) + '  ' + name + '\n' for name in ops.NATIVE))
        (folder / 'COMPLETE').touch()

    def files(self, objects, filename, volume=None):
        file_tar(filename, {objects[0]['physical']: b'abc'} if objects is not None else {'heartbeat': b'synthetic'}, filesystem=True)

    def current_storage(self):
        return inventory(version='generation-b') if self.changed else inventory()

    def private_config(self, stage):
        (stage / 'containers.private.json').write_bytes(ops.canonical(self.rows))
        (stage / 'arborscan.env.private').write_text(PRIVATE)
        file_tar(stage / 'local-files.tar', {'selfhost-root/compose.private.json': b'{}'})


def assert_prior_preserved(env):
    for name, raw in env.old_bytes.items():
        assert (env.output / name).read_bytes() == raw


def test_full_orchestration_atomic_marker_and_prior14_preserved(environment):
    FakeBackend.events = []
    result = ops.backup(environment.root, environment.output, environment.index,
        layout='version-directory', stamp='20261012T170000Z', factory=FakeBackend,
        free=lambda p: SimpleNamespace(free=10 * 1024**3))
    folder = environment.output / result['name']
    assert result['cloud_requests'] == 0
    assert result['rotation_performed'] is False
    assert result['database_restore_test_performed'] is False
    assert (folder / 'COMPLETE').is_file()
    assert not (folder / 'COMPLETE.partial').exists()
    assert ops.verify_set(folder) == result['verified_files']
    assert json.loads((folder / 'SELFHOST_BACKUP.json').read_text())['database_and_json_snapshot'] == 'same_exported_snapshot'
    assert FakeBackend.events == ['snapshot-open', ('native', '00000003-0000001B-1'), 'snapshot-close']
    assert len(list(environment.output.glob('*/COMPLETE'))) == 15
    assert_prior_preserved(environment)


@pytest.mark.parametrize('failure', ['storage_changed', 'native_error', 'runtime_checksum'])
def test_failures_cannot_publish_or_rotate(environment, failure):
    class Backend(FakeBackend):
        changed = failure == 'storage_changed'
        native_failure = failure == 'native_error'
    if failure == 'runtime_checksum':
        environment.asset.write_bytes(b'corrupted')
    with pytest.raises(ops.BackupError):
        ops.backup(environment.root, environment.output, environment.index,
            layout='version-directory', stamp='20261012T170000Z', factory=Backend,
            free=lambda p: SimpleNamespace(free=10 * 1024**3))
    assert not (environment.output / '20261012T170000Z').exists()
    assert_prior_preserved(environment)


def test_low_space_and_existing_timestamp_preserve_all_copies(environment):
    for stamp, space in [('20261012T170000Z', 1), ('20260901T030000Z', 10 * 1024**3)]:
        with pytest.raises(ops.BackupError):
            ops.backup(environment.root, environment.output, environment.index,
                layout='version-directory', stamp=stamp, factory=FakeBackend,
                free=lambda p: SimpleNamespace(free=space))
    assert_prior_preserved(environment)


def test_created_stopped_worker_and_private_bind_are_supported(environment, monkeypatch):
    root = environment.root
    bind = root / 'assets/model-quality'; bind.mkdir(parents=True)
    (root / 'compose.private.json').write_text(json.dumps({'services': {s: {} for s in ops.SERVICES}}))
    rows = FakeBackend(root, root).rows
    for row in rows:
        service = row['Config']['Labels']['com.docker.compose.service']
        row['Id'] = 'fixture-' + service
        row['State'] = {'Running': service in {'db', 'rest', 'storage', 'gateway'}}
        row['Mounts'] = []
        if service == 'storage':
            row['Config']['Env'] = ['STORAGE_BACKEND=file', 'FILE_STORAGE_BACKEND_PATH=/var/lib/storage',
                                     'TENANT_ID=stub', 'GLOBAL_S3_BUCKET=global']
            row['Mounts'] = [{'Type': 'volume', 'Name': 'fixture_storage', 'Destination': '/var/lib/storage'}]
        if service == 'worker':
            row['Mounts'] = [{'Type': 'bind', 'Source': str(bind), 'Destination': '/app/model-quality'}]
    commands = []
    def run(self, cmd, **kw):
        commands.append(cmd)
        return b'fixture-ids' if cmd[1] == 'compose' else ops.canonical(rows)
    monkeypatch.setattr(ops.DockerBackend, 'run', run)
    backend = ops.DockerBackend(root, root)
    assert backend.runtime_mount == {'type': 'bind', 'source': str(bind)}
    assert commands[0][-4:] == ['--profile', 'manual-worker', 'ps', '-aq']
    assert all('up' not in c and 'start' not in c for c in commands)
    backend.files(None, root / 'runtime-test.tar', backend.runtime_mount)
    assert 'type=bind,src=' + str(bind) + ',dst=/source,readonly' in commands[-1]
    assert '--network' in commands[-1] and 'none' in commands[-1]


def test_runtime_index_exports_only_missing_local_image_and_never_rewrites_old(environment):
    config = b'{}'; new_image = 'sha256:' + hashlib.sha256(config).hexdigest()
    compose = {'services': {s: {'image': IMAGE if s in {'v3', 'api', 'worker'} else new_image} for s in ops.SERVICES}}
    (environment.root / 'compose.private.json').write_text(json.dumps(compose))
    previous = environment.index.read_bytes()
    commands = []
    def run(cmd, *, stdout, stderr, timeout):
        commands.append(cmd)
        if 'inspect' in cmd:
            return SimpleNamespace(returncode=0, stdout=(cmd[3] + '\n').encode())
        members = {'config.json': config, 'layer.tar': b'synthetic-layer',
                   'manifest.json': ops.canonical([{'Config': 'config.json', 'Layers': ['layer.tar']}])}
        with tarfile.open(fileobj=stdout, mode='w') as archive:
            for name, raw in members.items():
                info = tarfile.TarInfo(name); info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
        return SimpleNamespace(returncode=0, stdout=None)
    directory = ops.PRIVATE_HOME / 'ops-runtime' / environment.root.name
    result = ops.create_runtime_index(environment.root, directory, environment.index, run=run,
        free=lambda _: SimpleNamespace(free=10 * 1024**3))
    assert result['images_exported'] == 1
    assert result['services'] == 7 and result['unique_images'] == 2
    assert environment.index.read_bytes() == previous
    published = (directory / 'RUNTIME_INDEX.json').read_bytes()
    second = ops.create_runtime_index(environment.root, directory, environment.index, run=run)
    assert second['images_exported'] == 0
    assert sum('save' in cmd for cmd in commands) == 1
    assert (directory / 'RUNTIME_INDEX.json').read_bytes() == published
    assert all('pull' not in cmd and 'start' not in cmd for cmd in commands)


def test_saved_image_config_must_match_actual_image_id(tmp_path):
    archive = tmp_path / 'wrong.tar'
    file_tar(archive, {'config.json': b'{}', 'layer.tar': b'fixture'})
    with tarfile.open(archive, 'a') as stream:
        raw = ops.canonical([{'Config': 'config.json', 'Layers': ['layer.tar']}])
        info = tarfile.TarInfo('manifest.json'); info.size = len(raw); stream.addfile(info, io.BytesIO(raw))
    with pytest.raises(ops.BackupError, match='image_id_mismatch'):
        ops.verify_saved_image(archive, IMAGE)


@pytest.mark.parametrize('corrupt_layer', [False, True])
def test_containerd_oci_index_id_has_verified_local_amd64_closure(tmp_path, corrupt_layer):
    config, layer = b'{}', b'synthetic-compressed-layer'
    config_sha, layer_sha = hashlib.sha256(config).hexdigest(), hashlib.sha256(layer).hexdigest()
    manifest = ops.canonical({'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.manifest.v1+json',
        'config': {'digest': 'sha256:' + config_sha}, 'layers': [{'digest': 'sha256:' + layer_sha}]})
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    index = ops.canonical({'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.index.v1+json',
        'manifests': [{'digest': 'sha256:' + manifest_sha, 'platform': {'os': 'linux', 'architecture': 'amd64'}},
                      {'digest': 'sha256:' + 'f' * 64, 'platform': {'os': 'linux', 'architecture': 'arm64'}}]})
    image = 'sha256:' + hashlib.sha256(index).hexdigest()
    members = {'blobs/sha256/' + config_sha: config,
               'blobs/sha256/' + layer_sha: b'corrupt' if corrupt_layer else layer,
               'blobs/sha256/' + manifest_sha: manifest,
               'blobs/sha256/' + image.removeprefix('sha256:'): index,
               'manifest.json': ops.canonical([{'Config': 'blobs/sha256/' + config_sha,
                   'Layers': ['blobs/sha256/' + layer_sha]}])}
    path = tmp_path / 'oci.tar'
    with tarfile.open(path, 'w') as archive:
        for name, raw in members.items():
            row = tarfile.TarInfo(name); row.size = len(raw); archive.addfile(row, io.BytesIO(raw))
    if corrupt_layer:
        with pytest.raises(ops.BackupError, match='layer_checksum_mismatch'):
            ops.verify_saved_image(path, image)
    else:
        ops.verify_saved_image(path, image)


def test_native_command_uses_exact_snapshot_and_actual_list_input(tmp_path):
    backend = ops.DockerBackend.__new__(ops.DockerBackend)
    backend.services = {'db': {'Id': 'synthetic-local-db'}}
    calls = []
    def run(command, *, output=None, stdin=None, **kwargs):
        calls.append(command)
        if 'pg_dump' in command:
            output.write(b'PGDMPsynthetic')
        elif 'pg_dumpall' in command:
            output.write(b'-- password-free roles')
        elif 'pg_restore' in command:
            assert stdin.read() == b'PGDMPsynthetic'
            output.write(b'actual-input-list')
        else:
            return b'{"server_version":"17.6","source":"local_selfhost"}'
    backend.run = run
    backend.native('00000003-0000001B-1', tmp_path / 'postgres')
    dump = next(command for command in calls if 'pg_dump' in command)
    assert '--snapshot=00000003-0000001B-1' in dump
    assert '--no-role-passwords' in next(command for command in calls if 'pg_dumpall' in command)
    assert (tmp_path / 'postgres/archive-list.private.txt').read_bytes() == b'actual-input-list'


def test_psql_driver_reads_live_process_and_rolls_back_without_output(tmp_path, capsys):
    driver = r'''
import base64,json,sys
def reply(value):print(base64.b64encode(json.dumps(value,indent=2).encode()).decode(),flush=True)
for line in sys.stdin:
 if 'pg_export_snapshot' in line: print('00000003-0000001B-1',flush=True)
 elif "json_build_object('objects'" in line: reply({'objects':[],'buckets':[]})
 elif 'FROM pg_class' in line: reply(['auth_sessions'])
 elif 'FROM public.' in line: reply([{'token':'synthetic-private-not-for-output'}])
 elif 'ROLLBACK' in line: break
'''
    backend = SimpleNamespace(log=tmp_path / 'private.log', psql=lambda: [sys.executable, '-u', '-c', driver])
    snapshot = ops.PsqlSnapshot(backend)
    try:
        assert snapshot.identifier == '00000003-0000001B-1'
        assert snapshot.storage() == {'objects': [], 'buckets': []}
        assert snapshot.tables() == {'auth_sessions': [{'token': PRIVATE}]}
    finally:
        snapshot.close()
    assert snapshot.process.returncode == 0
    assert capsys.readouterr().out == ''
    assert (tmp_path / 'private.log').read_bytes() == b''


def test_psql_frames_pretty_nested_json_and_unicode_in_one_response():
    import base64
    import queue
    snapshot = object.__new__(ops.PsqlSnapshot)
    snapshot.lines = queue.Queue()
    data = {'metadata': {'description': 'Снимок\nи разметка'}, 'objects': []}
    raw = json.dumps(data, ensure_ascii=False, indent=2).encode()
    snapshot.lines.put(base64.b64encode(raw).decode())
    sent = []; snapshot.send = sent.append
    assert snapshot.query("SELECT json_build_object('metadata', metadata);") == data
    assert 'replace(encode(convert_to((' in sent[0]
    assert "E'\\n', ''" in sent[0]
    snapshot.lines.put('not base64')
    with pytest.raises(ops.BackupError, match='invalid_snapshot_response'):
        snapshot.query('SELECT true;')


def test_psql_driver_rejects_early_connection_exit(tmp_path):
    backend = SimpleNamespace(log=tmp_path / 'private.log', psql=lambda: [sys.executable, '-c', 'pass'])
    with pytest.raises(ops.BackupError, match='connection_failed'):
        ops.PsqlSnapshot(backend)


def test_cli_failure_does_not_print_exception_or_private_contents(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['backup', '--root', '/private', '--runtime-index', '/private',
                                    '--storage-layout', 'unversioned'])
    monkeypatch.setattr(ops, 'backup', lambda *a, **k: (_ for _ in ()).throw(ops.BackupError(PRIVATE)))
    with pytest.raises(SystemExit) as failure:
        ops.main()
    assert failure.value.code == 1
    output = capsys.readouterr()
    assert output.out == ''
    assert PRIVATE not in output.err
    # A late fsync/receipt error can follow publication; generic CLI failure
    # must not falsely promise that no durable COMPLETE has been published.
    assert json.loads(output.err)['complete_published'] is None


def test_scheduled_failure_does_not_claim_rotation_or_publication_was_absent(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['backup', '--scheduled', '--root', '/private',
        '--runtime-index', '/private', '--storage-layout', 'unversioned'])
    monkeypatch.setattr(ops, 'scheduled_backup', lambda *a, **k: (_ for _ in ()).throw(ops.BackupError(PRIVATE)))
    with pytest.raises(SystemExit):
        ops.main()
    result = json.loads(capsys.readouterr().err)
    assert result['complete_published'] is None and result['prior_backups_modified'] is None
