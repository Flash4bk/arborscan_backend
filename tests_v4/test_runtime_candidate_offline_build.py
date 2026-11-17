"""Offline build orchestration at a fake Docker boundary; no daemon is used."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('offline_candidate', ROOT / 'deploy-vps/runtime_candidate.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
COMMIT = 'a' * 40
IMAGE = 'sha256:' + 'b' * 64
BUILT = 'sha256:' + 'c' * 64


def prepared_context(tmp_path):
    context = tmp_path / 'context'
    context.mkdir()
    files = {'config.py': b'# synthetic\n', 'arborscan_v4/legacy_feedback.py': b'# synthetic\n'}
    hashes = {}
    for name, body in files.items():
        path = context / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        hashes[name] = hashlib.sha256(body).hexdigest()
    runtime = {'image_id': IMAGE, 'python': '3.11.16', 'packages': {'fixture': '1.0'}}
    provenance = {'format': 1, 'role': 'v3', 'source_commit': COMMIT,
                  'base_image_id': IMAGE, 'source_sha256': hashes, 'built': False, 'deployed': False}
    (context / 'provenance.json').write_text(json.dumps(provenance))
    (context / 'Dockerfile').write_text(candidate.overlay_dockerfile('v3', runtime, COMMIT, sorted(files)))
    (context / '.dockerignore').write_text(candidate.context_ignore(files))
    return context, runtime


class DockerBoundary:
    def __init__(self, runtime):
        self.runtime = runtime
        self.images = {IMAGE: {'Id': IMAGE, 'Config': {'User': 'arborscan', 'WorkingDir': '/app',
                        'Volumes': None, 'Env': ['PATH=/old', 'SUPABASE_SERVICE_KEY=synthetic-private']},
                        'RepoTags': ['original:synthetic'], 'RepoDigests': []}}
        self.calls = []
        self.actual_calls = []
        self.environments = []
        self.driver = 'docker'
        self.fail_build = False
        self.mutate_alias = False
        self.mutate_references = False
        self.bad_inventory = False
        self.fail_imports = False
        self.private_config = False
        self.effective = None
        self.alias = None
        self.removed = []

    def run(self, args, **kwargs):
        args = list(args)
        self.actual_calls.append(args)
        self.environments.append(kwargs.get('env', {}))
        if args[:3] == ['docker', '--context', 'default']:
            args = ['docker', *args[3:]]
        self.calls.append(args)
        output, code = '', 0
        if args[:3] == ['docker', 'buildx', 'inspect']:
            output = json.dumps({'Name': 'default', 'Driver': self.driver, 'Nodes': [
                {'Name': 'default', 'Endpoint': 'default', 'Status': 'running'}]})
        elif args[:3] == ['docker', 'image', 'inspect']:
            row = self.images.get(args[3])
            if row is None:
                code = 1
            else:
                output = row['Id'] if args[-1] == '{{.Id}}' else json.dumps(row)
        elif args[:3] == ['docker', 'image', 'tag']:
            self.alias = args[-1]
            self.images[self.alias] = copy.deepcopy(self.images[args[-2]])
        elif args[:2] == ['docker', 'build'] or args[:3] == ['docker', 'buildx', 'build']:
            self.effective = Path(args[args.index('-f') + 1]).read_bytes().decode()
            # BuildKit treats a literal config ID as repository `sha256:<tag>`.
            if self.effective.startswith('FROM sha256:'):
                raise RuntimeError('would_resolve_registry')
            if self.mutate_alias:
                self.images[self.alias]['Id'] = 'sha256:' + 'd' * 64
            if self.mutate_references:
                self.images[self.alias]['RepoTags'] = [self.alias]
            if self.fail_build:
                raise subprocess.CalledProcessError(1, args)
            self.images[args[args.index('-t') + 1]] = copy.deepcopy(self.images[IMAGE]) | {'Id': BUILT}
        elif args[:3] == ['docker', 'image', 'rm']:
            self.removed.append(args[-1])
            del self.images[args[-1]]
        elif args[:2] == ['docker', 'run']:
            code_text = args[-1]
            if code_text == candidate.INVENTORY_CODE:
                row = copy.deepcopy(self.runtime)
                row['private_configuration_files'] = int(self.private_config)
                if self.bad_inventory:
                    row['packages'] = {}
                output = json.dumps(row)
            elif self.fail_imports and 'import cv2' in code_text:
                raise subprocess.CalledProcessError(1, args)
        else:
            raise AssertionError('Unexpected Docker boundary command: ' + repr(args))
        result = subprocess.CompletedProcess(args, code, stdout=output, stderr='')
        if code and kwargs.get('check'):
            raise subprocess.CalledProcessError(code, args)
        return result


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    context, runtime = prepared_context(tmp_path)
    docker = DockerBoundary(runtime)
    monkeypatch.setattr(candidate.subprocess, 'run', docker.run)
    monkeypatch.setattr(candidate, 'secrets', SimpleNamespace(token_hex=lambda _: '123456789abc'), raising=False)
    return context, runtime, docker


def test_buildkit_receives_local_alias_and_keeps_original_context_exact(boundary):
    context, runtime, docker = boundary
    original = {p.relative_to(context).as_posix(): p.read_bytes() for p in context.rglob('*') if p.is_file()}
    assert candidate.build_local(context, runtime, 'v3') == BUILT
    assert docker.effective.splitlines()[0] == 'FROM ' + docker.alias
    assert docker.effective.splitlines()[1:] == original['Dockerfile'].decode().splitlines()[1:]
    assert {p.relative_to(context).as_posix(): p.read_bytes() for p in context.rglob('*') if p.is_file()} == original
    assert docker.removed == [docker.alias]
    assert IMAGE in docker.images


@pytest.mark.parametrize('driver', ['docker-container', 'remote', 'kubernetes'])
def test_nonlocal_driver_fails_before_tagging_or_building(boundary, driver):
    context, runtime, docker = boundary
    docker.driver = driver
    with pytest.raises(ValueError, match='builder'):
        candidate.build_local(context, runtime, 'v3')
    assert not any(c[:3] == ['docker', 'image', 'tag'] for c in docker.calls)
    assert docker.effective is None


def test_all_docker_operations_pin_default_context_and_remove_remote_overrides(boundary, monkeypatch):
    context, runtime, docker = boundary
    for key in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'BUILDX_BUILDER'):
        monkeypatch.setenv(key, 'synthetic-other-endpoint')
    candidate.build_local(context, runtime, 'v3')
    assert all(args[:3] == ['docker', '--context', 'default'] for args in docker.actual_calls)
    assert all(not any(k.startswith(('DOCKER_', 'BUILDX_')) for k in env) for env in docker.environments)


@pytest.mark.parametrize('problem', ['missing', 'changed-id', 'declared-volume', 'no-original-reference'])
def test_unexpected_base_refuses_build_before_own_alias_creation(boundary, problem):
    context, runtime, docker = boundary
    if problem == 'missing':
        del docker.images[IMAGE]
    elif problem == 'changed-id':
        docker.images[IMAGE]['Id'] = 'sha256:' + 'd' * 64
    elif problem == 'declared-volume':
        docker.images[IMAGE]['Config']['Volumes'] = {'/private': {}}
    else:
        docker.images[IMAGE]['RepoTags'] = []
    with pytest.raises(ValueError, match='base|image'):
        candidate.build_local(context, runtime, 'v3')
    assert docker.effective is None and docker.alias is None


def test_build_failure_removes_only_own_alias_without_force_or_image_id(boundary):
    context, runtime, docker = boundary
    docker.fail_build = True
    with pytest.raises(subprocess.CalledProcessError):
        candidate.build_local(context, runtime, 'v3')
    assert docker.removed == [docker.alias]
    cleanup = [c for c in docker.calls if c[:3] == ['docker', 'image', 'rm']]
    assert cleanup == [['docker', 'image', 'rm', '--no-prune', docker.alias]]
    assert IMAGE in docker.images


@pytest.mark.parametrize('existing', ['base-alias', 'candidate'])
def test_preexisting_output_tag_is_never_replaced_or_removed(boundary, existing):
    context, runtime, docker = boundary
    tag = ('arborscan-runtime-base:v3-123456789abc' if existing == 'base-alias' else
           'arborscan-v3:local-candidate-' + COMMIT[:12] + '-' + IMAGE[7:19] + '-123456789abc')
    docker.images[tag] = copy.deepcopy(docker.images[IMAGE]) | {'Id': 'sha256:' + 'd' * 64}
    with pytest.raises(ValueError, match='existing'):
        candidate.build_local(context, runtime, 'v3')
    assert docker.images[tag]['Id'] == 'sha256:' + 'd' * 64
    assert docker.removed == [] and docker.alias is None


def test_changed_alias_is_preserved_and_never_used_for_smoke(boundary):
    context, runtime, docker = boundary
    docker.mutate_alias = True
    with pytest.raises(ValueError, match='alias'):
        candidate.build_local(context, runtime, 'v3')
    assert docker.removed == []
    assert not any(c[:2] == ['docker', 'run'] for c in docker.calls)


def test_alias_is_preserved_if_original_image_reference_disappeared(boundary):
    context, runtime, docker = boundary
    docker.mutate_references = True
    with pytest.raises(ValueError, match='reference'):
        candidate.build_local(context, runtime, 'v3')
    assert docker.removed == [] and docker.alias in docker.images


def test_all_helpers_are_isolated_and_clear_inherited_private_config(boundary):
    context, runtime, docker = boundary
    assert candidate.build_local(context, runtime, 'v3') == BUILT
    runs = [c for c in docker.calls if c[:2] == ['docker', 'run']]
    assert len(runs) == 3
    for args in runs:
        for flag in ('--rm', '--network=none', '--pull=never', '--read-only'):
            assert flag in args
        for flag, value in (('--memory', '2g'), ('--cpus', '1'), ('--pids-limit', '128'),
                            ('--cap-drop', 'ALL'), ('--security-opt', 'no-new-privileges')):
            assert args[args.index(flag) + 1] == value
        assert 'size=134217728' in args[args.index('--tmpfs') + 1]
        assert '--mount' not in args and '-v' not in args and '--env-file' not in args
        env = [args[i + 1] for i, value in enumerate(args) if value == '--env']
        assert 'SUPABASE_SERVICE_KEY=' in env
        assert 'PYTHONPYCACHEPREFIX=/tmp/as14-pycache' in env
        assert 'PYTHONDONTWRITEBYTECODE=1' in env
        assert 'YOLO_OFFLINE=true' in env and 'YOLO_AUTOINSTALL=false' in env
        assert all('synthetic-private' not in value for value in env)
        assert '-B' in args


def test_gate_executes_synthetic_cpu_native_ops_without_starting_api_or_models(boundary):
    context, runtime, docker = boundary
    candidate.build_local(context, runtime, 'v3')
    source = [c[-1] for c in docker.calls if c[:2] == ['docker', 'run'] and 'import cv2' in c[-1]][0]
    assert 'torch.add(' in source and 'torchvision.ops.nms(' in source and 'cv2.cvtColor(' in source
    assert 'server' not in source and 'arborscan_v4.api' not in source and 'YOLO(' not in source


def test_source_gate_accepts_generated_pycache_but_refuses_stale_python_module(boundary):
    context, runtime, docker = boundary
    candidate.build_local(context, runtime, 'v3')
    source = [c[-1] for c in docker.calls if c[:2] == ['docker', 'run']
              and c[-1].startswith('import hashlib,json,pathlib')][0]
    # Only remap the image root; execute the actual gate body unchanged.
    source = source.replace("'/app'", repr(str(context)))
    cache = context / 'arborscan_v4/__pycache__'
    cache.mkdir()
    (cache / 'generated.cpython-311.pyc').write_bytes(b'synthetic bytecode')
    exec(source, {})
    (context / 'arborscan_v4/stale.py').write_text('# stale image source\n')
    with pytest.raises(AssertionError, match='stale_source'):
        exec(source, {})


def test_private_dotenv_filename_in_image_refuses_all_library_imports(boundary):
    context, runtime, docker = boundary
    docker.private_config = True
    with pytest.raises(ValueError, match='private_configuration'):
        candidate.build_local(context, runtime, 'v3')
    runs = [c for c in docker.calls if c[:2] == ['docker', 'run']]
    assert len(runs) == 1 and runs[0][-1] == candidate.INVENTORY_CODE
    assert docker.removed == [docker.alias]


@pytest.mark.parametrize('failure', ['bad_inventory', 'fail_imports'])
def test_gate_failure_never_reports_success_and_still_cleans_own_alias(boundary, failure):
    context, runtime, docker = boundary
    setattr(docker, failure, True)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        candidate.build_local(context, runtime, 'v3')
    assert docker.removed == [docker.alias]
    assert json.loads((context / 'provenance.json').read_text())['built'] is False


def test_audit_records_recipe_hash_alias_and_env_key_names_only(boundary):
    context, runtime, docker = boundary
    audit = {}
    candidate.build_local(context, runtime, 'v3', audit=audit)
    assert audit['base_image_id'] == IMAGE and audit['base_alias'] == docker.alias
    assert audit['effective_dockerfile_sha256'] == hashlib.sha256(docker.effective.encode()).hexdigest()
    assert audit['inherited_environment_keys'] == ['PATH', 'SUPABASE_SERVICE_KEY']
    assert 'synthetic-private' not in json.dumps(audit)
    assert audit['base_alias_removed'] is True


def test_modified_context_fails_before_any_docker_command(boundary):
    context, runtime, docker = boundary
    (context / 'config.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='context'):
        candidate.build_local(context, runtime, 'v3')
    assert docker.calls == []
