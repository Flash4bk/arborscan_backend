"""No Docker daemon required: reject mutable/incomplete release inputs locally."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('runtime_candidate', ROOT / 'deploy-vps/runtime_candidate.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def manifest():
    return json.loads((ROOT / 'deploy-vps/runtime-lock.json').read_text())


def test_running_image_manifest_is_complete_and_separates_duplicate_opencv():
    value = candidate.validate_manifest(manifest())
    for role in ('v3', 'v4'):
        row = value['runtimes'][role]
        assert row['image_id'].startswith('sha256:') and len(row['image_id']) == 71
        assert row['python'] == '3.11.16'
        assert len(row['packages']) >= 50
        assert 'opencv-python' in row['packages']
        assert 'opencv-python-headless' in row['packages']


@pytest.mark.parametrize('base', ['arborscan-api:latest', 'python:3.11', 'sha256:' + 'a'*63])
def test_mutable_or_malformed_base_is_never_a_release_runtime(base):
    value = manifest(); value['runtimes']['v3']['image_id'] = base
    with pytest.raises(ValueError, match='image'):
        candidate.validate_manifest(value)


def test_selected_runtime_rejects_missing_or_changed_packages_and_python():
    row = manifest()['runtimes']['v4']
    assert candidate.verify_inventory(row, {k: row[k] for k in ('python', 'packages')}) == []
    for field in ('python', 'packages'):
        observed = {k: row[k] for k in ('python', 'packages')}
        observed[field] = '0.0' if field == 'python' else {'fastapi': '0.0'}
        assert candidate.verify_inventory(row, observed)


def test_overlay_closure_includes_science_and_owner_feedback_without_installing_packages():
    files = candidate.required_sources(ROOT, 'v4')
    assert 'arborscan_science/section.py' in files
    assert 'arborscan_v4/science_api.py' in files
    assert 'arborscan_v4/safe_logging.py' in files
    v3 = candidate.required_sources(ROOT, 'v3')
    assert 'arborscan_v4/legacy_feedback.py' in v3
    assert 'arborscan_v4/corrections_api.py' in v3
    docker = candidate.overlay_dockerfile('v4', manifest()['runtimes']['v4'], 'a'*40, files)
    assert 'pip install' not in docker and 'apt-get' not in docker
    assert 'arborscan_science/section.py' in docker
    assert 'sha256:2abb56ed' in docker


def test_incomplete_source_tree_fails_before_a_candidate_is_created(tmp_path):
    with pytest.raises(ValueError, match='source'):
        candidate.required_sources(tmp_path, 'v3')


@pytest.mark.parametrize('name', ['/outside.py', '../outside.py', 'package/../../outside.py'])
def test_overlay_rejects_absolute_or_parent_source_paths(name):
    with pytest.raises(ValueError, match='source'):
        candidate.overlay_dockerfile('v4', manifest()['runtimes']['v4'], 'a'*40, [name])


def test_clean_build_versions_are_pinned_and_only_one_cv2_provider_is_declared():
    for role in ('v3', 'v4'):
        text = (ROOT / f'deploy-vps/requirements.runtime-{role}.lock.txt').read_text()
        rows = [line for line in text.splitlines() if line and not line.startswith('#')]
        assert all('==' in line and '://' not in line for line in rows)
        assert len(rows) > 45
        assert len([line for line in rows if line.startswith('opencv-')]) == 1
        assert next(line for line in rows if line.startswith('opencv-')).startswith('opencv-python==')
        assert next(line for line in rows if line.startswith('torch==')).endswith('+cpu')


def test_context_destination_cannot_reuse_an_existing_directory_or_source_tree(tmp_path):
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(ValueError):
        candidate.validate_destination(ROOT, ROOT)
    with pytest.raises(ValueError):
        candidate.validate_destination(ROOT, existing)


def synthetic_repository(path):
    path.mkdir()
    for name in ['config.py', 'server.py', 'auth_google_identity.py', 'tree_dynamics.py',
                 'arborscan_v4/safe_logging.py', 'arborscan_v4/corrections_api.py',
                 'arborscan_v4/legacy_feedback.py', 'arborscan_v4/science_api.py',
                 'arborscan_science/section.py']:
        file = path / name; file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('# committed synthetic source\n')
    (path / '.env').write_text('TEST_ONLY=synthetic-private-fixture\n')
    for args in [('init', '-q'), ('add', '.'),
                 ('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                  'commit', '-qm', 'Synthetic source closure')]:
        subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True)
    return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_preparation_uses_exact_commit_ignores_dirty_source_and_never_copies_env(tmp_path):
    repo = tmp_path / 'repo'; commit = synthetic_repository(repo)
    (repo / 'server.py').write_text('# later uncommitted source\n')
    context = repo / 'output' / 'candidate'
    result = candidate.prepare_context(repo, 'v3', commit, context, manifest())
    assert (context / 'server.py').read_text() == '# committed synthetic source\n'
    assert not (context / '.env').exists()
    assert result['built'] is False and result['deployed'] is False
    assert result['source_commit'] == commit
    assert '.env' not in result['source_sha256']


def test_missing_new_module_in_selected_commit_is_not_filled_from_working_tree(tmp_path):
    repo = tmp_path / 'repo'; commit = synthetic_repository(repo)
    subprocess.run(['git', '-C', str(repo), 'rm', '-q', 'arborscan_v4/legacy_feedback.py'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'Incomplete closure'], check=True, capture_output=True)
    incomplete = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
                                 capture_output=True, text=True).stdout.strip()
    (repo / 'arborscan_v4/legacy_feedback.py').write_text('# uncommitted replacement\n')
    context = repo / 'output' / 'candidate'
    with pytest.raises(ValueError, match='source'):
        candidate.prepare_context(repo, 'v3', incomplete, context, manifest())
    assert not context.exists()


def test_prepared_context_rejects_modified_source_and_extra_files_before_docker(tmp_path):
    repo = tmp_path / 'repo'; commit = synthetic_repository(repo)
    context = repo / 'output' / 'candidate'
    candidate.prepare_context(repo, 'v3', commit, context, manifest())
    candidate.verify_context(context, manifest()['runtimes']['v3'], 'v3')
    original = (context / 'server.py').read_bytes()
    (context / 'server.py').write_text('# accidental unreviewed edit\n')
    with pytest.raises(ValueError, match='context'):
        candidate.verify_context(context, manifest()['runtimes']['v3'], 'v3')
    (context / 'server.py').write_bytes(original)
    (context / 'arborscan_v4' / '.env').write_text('SYNTHETIC=not-a-real-secret')
    with pytest.raises(ValueError, match='context'):
        candidate.verify_context(context, manifest()['runtimes']['v3'], 'v3')


def test_prepared_context_rejects_modified_dockerfile_before_docker(tmp_path):
    repo = tmp_path / 'repo'; commit = synthetic_repository(repo)
    context = repo / 'output' / 'candidate'
    candidate.prepare_context(repo, 'v3', commit, context, manifest())
    (context / 'Dockerfile').write_text('FROM mutable:latest\n')
    with pytest.raises(ValueError, match='context'):
        candidate.verify_context(context, manifest()['runtimes']['v3'], 'v3')


def test_native_clean_build_is_explicitly_gated_not_declared_equivalent_to_overlay():
    for name in ['Dockerfile', 'Dockerfile.v4']:
        docker = (ROOT / 'deploy-vps' / name).read_text(encoding='utf-8')
        assert 'ARG PYTHON_BASE_IMAGE' in docker
        assert 'FROM ${PYTHON_BASE_IMAGE}' in docker
        assert 'python -m pip check' in docker
        assert '--no-deps torch==2.14.0+cpu torchvision==0.29.0+cpu' in docker
        assert 'importlib.metadata' in docker and 'opencv-python-headless' in docker
