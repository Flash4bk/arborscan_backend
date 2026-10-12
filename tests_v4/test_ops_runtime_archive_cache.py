"""Shared runtime optimizations must not weaken private backup integrity."""
import copy
import hashlib
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'deploy-vps'))
import ops_backup_policy as policy
import test_ops_backup_policy as fixtures


@pytest.fixture
def environment():
    fixture = fixtures.BackupPolicyTest()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.tearDown()


def count_archive_reads(monkeypatch, path):
    reads = []
    original = hashlib.file_digest
    identity = path.stat()
    def read(stream, algorithm):
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) == (identity.st_dev, identity.st_ino):
            reads.append(path)
        return original(stream, algorithm)
    monkeypatch.setattr(hashlib, 'file_digest', read)
    return reads


@pytest.mark.skipif(os.name == 'nt', reason='Runtime cache requires Linux mutation-sensitive ctime')
def test_shared_runtime_read_once_but_each_set_remains_verified(environment, monkeypatch):
    env = environment
    folders = env.existing(14)
    reads = count_archive_reads(monkeypatch, env.asset)
    rows = policy.inventory(env.root, runtime_cache=policy._RuntimeArchiveCache())
    assert len(rows) == 14 and all(row['valid'] for row in rows)
    assert len(reads) == 1
    # External callers have no implicit cache or persisted previous trust.
    policy.inventory(env.root)
    assert len(reads) == 15
    assert all((folder / 'COMPLETE').is_file() for folder in folders)


@pytest.mark.skipif(os.name == 'nt', reason='Runtime cache requires Linux mutation-sensitive ctime')
def test_cache_lifetime_is_one_locked_invocation(environment, monkeypatch):
    env = environment
    env.existing(13)
    reads = count_archive_reads(monkeypatch, env.asset)
    first = env.run_backup()
    assert first['event'] == 'completed_verified' and first['deleted'] == []
    assert len(reads) == 1
    second = env.run_backup()
    assert second['event'] == 'verified_existing'
    assert len(reads) == 2 and len(env.created) == 1


@pytest.mark.skipif(os.name == 'nt', reason='Runtime cache requires Linux mutation-sensitive ctime')
def test_immediate_deletion_candidate_rechecks_all_runtime_bytes(environment, monkeypatch):
    env = environment
    before = env.existing(14)
    reads = count_archive_reads(monkeypatch, env.asset)
    result = env.run_backup()
    assert result['deleted'] == [before[0].name]
    # One shared read during planning, one independent destructive-candidate
    # verification immediately before removal. No candidate uses the cache.
    assert len(reads) == 2
    assert not before[0].exists() and all(p.exists() for p in before[1:])


def test_cached_runtime_does_not_skip_per_set_file_or_native_integrity(environment):
    env = environment
    first, second = env.existing(2)
    cache = policy._RuntimeArchiveCache()
    policy.verify_set(first, runtime_cache=cache)
    (second / 'application.tar').write_bytes(b'damaged distinct set')
    with pytest.raises(policy.PolicyError, match='checksum_mismatch'):
        policy.verify_set(second, runtime_cache=cache)
    (first / 'postgres/database.dump').write_bytes(b'damaged native archive')
    with pytest.raises(policy.PolicyError, match='checksum_mismatch'):
        policy.verify_set(first, runtime_cache=cache)


def test_expected_digest_and_roles_still_checked_on_cached_reuse(environment):
    env = environment
    folder = env.existing(1)[0]
    cache = policy._RuntimeArchiveCache()
    policy.verify_runtime(folder, env.runtime, runtime_cache=cache)
    wrong = copy.deepcopy(env.runtime)
    wrong['images'][fixtures.API][0]['sha256'] = '0' * 64
    with pytest.raises(policy.PolicyError, match='runtime_archive_checksum_mismatch'):
        policy.verify_runtime(folder, wrong, runtime_cache=cache)
    wrong = copy.deepcopy(env.runtime)
    wrong['images'].pop(fixtures.WORKER)
    with pytest.raises(policy.PolicyError, match='runtime_index_image_mismatch'):
        policy.verify_runtime(folder, wrong, runtime_cache=cache)


@pytest.mark.skipif(os.name == 'nt', reason='Runtime cache requires Linux mutation-sensitive ctime')
def test_changed_file_is_refused_instead_of_reusing_a_previous_digest(environment):
    env = environment
    folder = env.existing(1)[0]
    cache = policy._RuntimeArchiveCache()
    policy.verify_runtime(folder, env.runtime, runtime_cache=cache)
    env.asset.write_bytes(b'changed archive with different size')
    with pytest.raises(policy.PolicyError, match='runtime_archive_changed_after_verification'):
        policy.verify_runtime(folder, env.runtime, runtime_cache=cache)


@pytest.mark.skipif(os.name == 'nt', reason='Runtime cache requires Linux mutation-sensitive ctime')
def test_atomic_replacement_cannot_reuse_even_identical_size_and_time(environment):
    env = environment
    cache = policy._RuntimeArchiveCache()
    cache.digest(env.asset)
    previous = env.asset.stat()
    replacement = env.asset.with_suffix('.replacement')
    replacement.write_bytes(b'x' * previous.st_size)
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    os.replace(replacement, env.asset)
    with pytest.raises(policy.PolicyError, match='runtime_archive_changed_after_verification'):
        cache.digest(env.asset)


@pytest.mark.skipif(os.name == 'nt', reason='Linux runtime file mode check')
def test_permissions_change_cannot_reuse_a_successful_digest(environment):
    env = environment
    cache = policy._RuntimeArchiveCache()
    cache.digest(env.asset)
    previous_mode = env.asset.stat().st_mode
    env.asset.chmod(0o400)
    assert env.asset.stat().st_mode != previous_mode
    with pytest.raises(policy.PolicyError, match='runtime_archive_changed_after_verification'):
        cache.digest(env.asset)


@pytest.mark.skipif(os.name == 'nt', reason='Linux ctime detects in-place writes after mtime restoration')
def test_in_place_same_size_write_with_restored_mtime_changes_ctime(environment):
    env = environment
    cache = policy._RuntimeArchiveCache()
    cache.digest(env.asset)
    previous = env.asset.stat()
    env.asset.write_bytes(b'x' * previous.st_size)
    os.utime(env.asset, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert env.asset.stat().st_ctime_ns != previous.st_ctime_ns
    with pytest.raises(policy.PolicyError, match='runtime_archive_changed_after_verification'):
        cache.digest(env.asset)


@pytest.mark.skipif(os.name == 'nt', reason='Runtime cache requires Linux mutation-sensitive ctime')
def test_mutation_during_first_hash_does_not_enter_cache(environment, monkeypatch):
    env = environment
    cache = policy._RuntimeArchiveCache()
    original = hashlib.file_digest
    def read(stream, algorithm):
        result = original(stream, algorithm)
        env.asset.write_bytes(b'concurrent larger changed runtime archive bytes')
        return result
    monkeypatch.setattr(hashlib, 'file_digest', read)
    with pytest.raises(policy.PolicyError, match='runtime_archive_changed_during_verification'):
        cache.digest(env.asset)
    assert cache._verified == {}


def test_stat_fingerprint_covers_ownership_permissions_and_canonical_identity(environment):
    path = environment.asset
    info = path.stat()
    assert policy._RuntimeArchiveCache._state(path) == (
        str(path.resolve()), info.st_dev, info.st_ino, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns, info.st_mode, info.st_uid, info.st_gid)


@pytest.mark.skipif(os.name != 'nt', reason='Windows producer compatibility fallback')
def test_windows_does_not_cache_or_trust_creation_time(environment, monkeypatch):
    env = environment
    reads = count_archive_reads(monkeypatch, env.asset)
    cache = policy._RuntimeArchiveCache()
    original = cache.digest(env.asset)
    before = env.asset.stat()
    env.asset.write_bytes(b'x' * before.st_size)
    os.utime(env.asset, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cache.digest(env.asset) != original
    assert len(reads) == 2 and cache._verified == {}


def test_parent_symlink_is_refused_even_after_previous_success(tmp_path):
    parent = tmp_path / 'assets'
    parent.mkdir()
    path = parent / 'image.tar'
    path.write_bytes(b'image archive')
    cache = policy._RuntimeArchiveCache()
    cache.digest(path)
    renamed = tmp_path / 'renamed'
    parent.rename(renamed)
    try:
        parent.symlink_to(renamed, target_is_directory=True)
    except OSError:
        pytest.skip('Creating directory symlinks is unavailable in this environment')
    with pytest.raises(policy.PolicyError, match='symlink_refused'):
        cache.digest(path)
