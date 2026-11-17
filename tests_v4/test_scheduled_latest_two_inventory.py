"""Remote byte budgets use tiny fake payloads, never SSH or installed D tools."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_local_retention as compact
import ops_pull_backups as pull


def newer_reader_source():
    # Same newer signature/source boundary as the installed reader; fake I/O
    # keeps the test standalone. The transport body is an opaque sentinel.
    inventory = pull.INVENTORY.replace('protected=set()',
        "protected=set()\nids=subprocess.run(['docker','ps','-aq'],capture_output=True,text=True,check=True).stdout.split()")
    inventory = inventory.replace("['docker','inspect','arborscan-api','arborscan-api-v4','arborscan-quality-worker']",
                                  "['docker','inspect',*ids]")
    inventory = inventory.replace('runtime_checked={}',
        "runtime_checked={}\ntransition=root/'SELFHOST_TRANSITION_PROTECTION.private.json'\n"
        "if transition.is_symlink():raise ValueError('unsafe_transition_protection')\n"
        "protection=json.loads(transition.read_bytes()) if transition.is_file() else None")
    inventory = inventory.replace("  runtime_file=p/'RUNTIME_DEPENDENCIES.json'",
        "  if protection is not None:record['transition_protection']=protection\n"
        "  runtime_file=p/'RUNTIME_DEPENDENCIES.json'")
    return ('import json,subprocess\nINVENTORY = ' + repr(inventory) + '''
def read_inventory(command, *, hidden=None, log=None, run=None, sleep=None,
                   known=None, timeout=180):
    run = subprocess.run if run is None else run
    source = INVENTORY if not known else 'KNOWN_LOCAL_MANIFESTS = ' + repr(dict(sorted(known.items()))) + '\\n' + INVENTORY
    # Preserve retry/hostkey/timeout transport body exactly.
    result = run(command, input=source.encode(), capture_output=True, timeout=timeout, **(hidden or {}))
    return json.loads(result.stdout)
def main():
    try:
            records=read_inventory(command,hidden=hidden,log=log,
                                    known=known_local_manifests(root),timeout=args.inventory_timeout)
            def delta(record,filename,target):
                return preserve_existing_resume(record)
            preserve_transition_history(root, records)
            latest = records[-1]
    except Exception:
            raise
def opaque_verifier():
    return preserve_all_native_runtime_checks()
''')


class TrackedStream:
    def __init__(self, stream, path, counts):
        self.stream, self.path, self.counts = stream, path, counts

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def read(self, *args):
        value = self.stream.read(*args)
        self.counts[self.path] = self.counts.get(self.path, 0) + len(value)
        return value

    def readinto(self, buffer):
        size = self.stream.readinto(buffer)
        self.counts[self.path] = self.counts.get(self.path, 0) + (size or 0)
        return size


@pytest.fixture
def remote(tmp_path):
    root = tmp_path / 'remote'
    root.mkdir()
    names = []
    payloads = set()
    for day in range(1, 29):
        folder = root / f'202601{day:02}T030000Z'
        folder.mkdir()
        names.append(folder.name)
        files = {name: b'fixture' for name in
                 ('application.tar', 'local-files.tar', 'arborscan.env.private', 'containers.private.json',
                  'postgres/database.dump', 'postgres/roles.sql', 'postgres/source.json',
                  'postgres/archive-list.private.txt', 'postgres/COMPLETE')}
        native = {name.split('/', 1)[1]: raw for name, raw in files.items() if name.startswith('postgres/')
                  and not name.endswith('/COMPLETE')}
        files['postgres/SHA256SUMS'] = ''.join(hashlib.sha256(raw).hexdigest() + '  ' + name + '\n'
                                               for name, raw in native.items()).encode()
        asset = tmp_path / (folder.name + '.archive')
        asset.write_bytes(('image-' + folder.name).encode())
        payloads.add(asset)
        files['RUNTIME_DEPENDENCIES.json'] = json.dumps({'format': 1, 'images': {
            'sha256:' + 'a' * 64: [{'path': str(asset), 'sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}]}}).encode()
        for name, raw in files.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        payloads.add(folder / 'application.tar')
        (folder / 'SHA256SUMS').write_text(''.join(hashlib.sha256(raw).hexdigest() + '  ' + name + '\n'
                                                  for name, raw in files.items()))
        (folder / 'COMPLETE').touch()
    protection = {'format': 1, 'sets': {name: 'historical-protection' for name in names[:14]}}
    (root / 'SELFHOST_TRANSITION_PROTECTION.private.json').write_text(json.dumps(protection))
    policy = {'format': 1, 'kind': 'owner_selected_local_latest_two', 'retain': 2,
              'authorization': 'explicit_owner_request'}
    return SimpleNamespace(root=root, names=names, payloads=payloads, policy=policy,
                           protection=protection, reads={}, docker=[], events=[])


def execute_main(remote, policy=True, source=None):
    source = compact.adapt_reader_source(source or newer_reader_source())
    namespace = {}
    exec(compile(source, '<adapted-newer-reader-fixture>', 'exec'), namespace)
    namespace.update(root=remote.root, command=['ssh', '-o', 'StrictHostKeyChecking=yes', 'fixture', 'python3 -'],
                     hidden={}, log=lambda *args, **kwargs: None,
                     known_local_manifests=lambda root: {}, args=SimpleNamespace(inventory_timeout=180, retention_dry_run=True),
                     preserve_transition_history=lambda *args: None)
    original_open = Path.open
    original_stat = Path.stat

    def counted_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return TrackedStream(stream, path, remote.reads)

    def virtual_stat(path, *args, **kwargs):
        info = original_stat(path, *args, **kwargs)
        if path.name != 'application.tar':
            return info
        class VirtualStat:
            st_size = 33 * 1024**2
            def __getattr__(self, name):
                return getattr(info, name)
        return VirtualStat()

    def run(command, **kwargs):
        if command[0] == 'docker':
            remote.docker.append(command)
            if command[1] == 'ps':
                return SimpleNamespace(stdout='all-container-one\nall-container-two\n')
            assert command == ['docker', 'inspect', 'all-container-one', 'all-container-two']
            return SimpleNamespace(stdout=json.dumps([{'Config': {'Labels': {}}, 'Mounts': [{
                'Source': '/home/arborscan/ops-backups/' + remote.names[0] + '/live-bind'}]}]))
        assert command == namespace['command']
        remote.events.append('inventory')
        script = kwargs['input'].decode().replace("root=pathlib.Path('/home/arborscan/ops-backups')",
                                                'root=pathlib.Path(' + repr(str(remote.root)) + ')')
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(script, '<fake-remote-inventory>', 'exec'), {})
        remote.last_inventory = json.loads(output.getvalue())
        return SimpleNamespace(stdout=output.getvalue().encode())

    selected = []
    def load(root):
        remote.events.append('policy')
        return remote.policy if policy else None
    def plan(root, policy, selected):
        namespace['selected_names'] = selected
        return {'before': 2, 'delete': []}
    with patch.object(compact, 'load_policy', side_effect=load), \
            patch.object(compact, 'plan', side_effect=plan), patch.object(compact, 'write_plan'), \
            patch.object(subprocess, 'run', side_effect=run), \
            patch.object(Path, 'open', counted_open), patch.object(Path, 'stat', virtual_stat):
        namespace['main']()
    return namespace


def test_policy_selects_before_historical_payload_reads(remote):
    namespace = execute_main(remote)
    assert namespace['selected_names'] == remote.names[-2:]
    heavy = {path for path in remote.payloads if remote.reads.get(path, 0)}
    assert len(heavy) == 4  # Two snapshot blockmaps and two full runtime hashes.
    assert sum(remote.reads[path] for path in remote.payloads if path in remote.reads) == sum(
        path.stat().st_size for path in heavy)
    assert remote.events == ['policy', 'inventory']
    assert remote.docker == [['docker', 'ps', '-aq'], ['docker', 'inspect', 'all-container-one', 'all-container-two']]
    assert all(row['protected_sets'] == [remote.names[0]] for row in remote.last_inventory)
    assert all(row['transition_protection'] == remote.protection for row in remote.last_inventory)


def test_policy_absent_preserves_remote_source_bytes_and_all_records(remote):
    namespace = execute_main(remote, policy=False)
    assert len({path for path in remote.payloads if remote.reads.get(path, 0)}) == 56
    original = ast.literal_eval(next(node.value for node in ast.parse(newer_reader_source()).body
                                    if isinstance(node, ast.Assign) and node.targets[0].id == 'INVENTORY'))
    assert compact.inventory_source(original, None) is original


@pytest.mark.parametrize('failure', ['missing-manifest', 'bad-sha', 'missing-runtime',
                                    'runtime-sha', 'empty-runtime', 'missing-native'])
def test_malformed_newest_modern_set_refuses_without_falling_back(remote, failure):
    folder = remote.root / remote.names[-1]
    manifest = folder / 'SHA256SUMS'
    if failure == 'missing-manifest':
        manifest.unlink()
    elif failure == 'bad-sha':
        manifest.write_text('not-a-sha  application.tar\n')
    elif failure in ('missing-runtime', 'missing-native'):
        marker = 'RUNTIME_DEPENDENCIES.json' if failure == 'missing-runtime' else 'postgres/database.dump'
        manifest.write_text(''.join(line + '\n' for line in manifest.read_text().splitlines()
                                    if not line.endswith('  ' + marker)))
    elif failure == 'runtime-sha':
        (folder / 'RUNTIME_DEPENDENCIES.json').write_text('{}')
    else:
        runtime = folder / 'RUNTIME_DEPENDENCIES.json'
        runtime.write_text(json.dumps({'format': 1, 'images': {}}))
        manifest.write_text(''.join((hashlib.sha256(runtime.read_bytes()).hexdigest() + '  RUNTIME_DEPENDENCIES.json'
                                    if line.endswith('  RUNTIME_DEPENDENCIES.json') else line) + '\n'
                                   for line in manifest.read_text().splitlines()))
    with pytest.raises((ValueError, FileNotFoundError)):
        execute_main(remote)
    assert not any(remote.reads.get(path, 0) for path in remote.payloads)


def test_newer_clearly_legacy_folder_can_be_skipped_and_older_corrupt_is_not_read(remote):
    legacy = remote.root / '20260201T030000Z'
    legacy.mkdir()
    files = {name: b'legacy' for name in ('application.tar', 'local-files.tar',
                                       'arborscan.env.private', 'containers.private.json')}
    for name, raw in files.items():
        (legacy / name).write_bytes(raw)
    (legacy / 'SHA256SUMS').write_text(''.join(hashlib.sha256(raw).hexdigest() + '  ' + name + '\n'
                                              for name, raw in files.items()))
    (legacy / 'COMPLETE').touch()
    old_manifest = remote.root / remote.names[0] / 'SHA256SUMS'
    old_manifest.write_text('old-corruption-must-not-be-read')
    namespace = execute_main(remote)
    assert namespace['selected_names'] == remote.names[-2:]
    assert not remote.reads.get(old_manifest, 0)


def test_adapter_is_idempotent_preserves_opaque_verifiers_and_refuses_partial_layout():
    original = newer_reader_source()
    adapted = compact.adapt_reader_source(original)
    assert compact.adapt_reader_source(adapted) == adapted
    assert 'return preserve_all_native_runtime_checks()' in adapted
    assert 'return preserve_existing_resume(record)' in adapted
    assert 'StrictHostKeyChecking' not in adapted  # Command/authentication remains caller-owned.
    assert adapted.index('compact_policy = load_policy(root)') < adapted.index('records=read_inventory(')
    mutations = [adapted.replace('inventory_source=compact_inventory_source', 'broken_hook=compact_inventory_source'),
                 adapted.replace('            records = select_records(records, compact_policy)\n', ''),
                 adapted + '\n# local_latest_two_inventory_v2\n']
    for changed in mutations:
        with pytest.raises(ValueError, match='unknown_installed_reader_layout'):
            compact.adapt_reader_source(changed)


def test_default_read_inventory_transports_identical_bytes_with_and_without_known_hints():
    original = {}
    adapted = {}
    exec(newer_reader_source(), original)
    exec(compact.adapt_reader_source(newer_reader_source()), adapted)
    for hints in (None, {'20260101T030000Z': 'a' * 64}):
        transmitted = []
        def run(command, **kwargs):
            transmitted.append(kwargs['input'])
            return SimpleNamespace(stdout=b'[]')
        original['read_inventory'](['ssh'], known=hints, run=run)
        adapted['read_inventory'](['ssh'], known=hints, run=run)
        assert transmitted[0] == transmitted[1]


@pytest.mark.parametrize('target', ['folder', 'COMPLETE', 'SHA256SUMS', 'application.tar', 'runtime-asset'])
def test_newest_symlink_refuses_before_any_heavy_reads(remote, target):
    folder = remote.root / remote.names[-1]
    path = (folder if target == 'folder' else next(path for path in remote.payloads
            if path.name == remote.names[-1] + '.archive') if target == 'runtime-asset' else folder / target)
    original = Path.is_symlink
    with patch.object(Path, 'is_symlink', lambda item: item == path or original(item)):
        with pytest.raises(ValueError, match='unsafe_latest_two'):
            execute_main(remote)
    assert not any(remote.reads.get(item, 0) for item in remote.payloads)


@pytest.mark.parametrize('target,limit', [('SHA256SUMS', 4 * 1024**2),
                                        ('postgres/SHA256SUMS', 1024**2),
                                        ('RUNTIME_DEPENDENCIES.json', 2 * 1024**2)])
def test_latest_metadata_size_is_bounded_before_heavy_reads(remote, target, limit):
    (remote.root / remote.names[-1] / target).write_bytes(b'x' * (limit + 1))
    with pytest.raises(ValueError, match='invalid_latest_two_metadata'):
        execute_main(remote)
    assert not any(remote.reads.get(item, 0) for item in remote.payloads)
