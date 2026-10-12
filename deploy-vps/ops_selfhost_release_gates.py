"""Read-only release gates bound to this private recovery and verified bytes.

No container, source database or user row is changed. A successful marker alone
cannot authorize cutover: retain and compare the captured source fingerprints,
the synthetic quarantine snapshot and the exact isolated restore receipt.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import re

import ops_selfhost_audit as audit
import ops_selfhost_backup as backup
import ops_selfhost_quarantine as quarantine


class GateFailure(Exception):
    """Fixed safe category; never include source data or credentials."""


def private_file(path, *, parent=None):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink()
            or any(item.is_symlink() for item in path.parents)
            or (parent is not None and path.parent != Path(parent))):
        raise GateFailure('release_evidence_path_differs')
    # Apply Linux permission checks even to non-JSON fingerprint inputs.
    if not path.is_file():
        raise GateFailure('release_evidence_missing')
    if audit.os.name != 'nt' and path.stat().st_mode & 0o077:
        raise GateFailure('release_evidence_not_private')
    return path


def source_gate(root, proof_path, *, now=None):
    proof_path = private_file(proof_path, parent=root)
    proof = audit.private_json(proof_path)
    try:
        captured = dt.datetime.strptime(proof['captured_at_utc'], '%Y%m%dT%H%M%SZ').replace(tzinfo=dt.timezone.utc)
    except (KeyError, TypeError, ValueError):
        raise GateFailure('fresh_source_timestamp_required') from None
    current = now or dt.datetime.now(dt.timezone.utc)
    age = (current - captured).total_seconds()
    if age < -60 or age > 120 * 60:
        raise GateFailure('fresh_source_proof_expired')
    if (proof.get('source_public_exact') is not True
            or proof.get('source_storage_exact') is not True
            or proof.get('read_only_single_snapshot') is not True
            or proof.get('sql_writes') is not False):
        raise GateFailure('actual_readonly_source_snapshot_required')
    expected = audit.private_json(root / 'audit.expected.private.json')
    source_public = audit.parse_fingerprints(private_file(proof.get('public_file', ''), parent=root).read_bytes())
    source_storage = audit.parse_storage_fingerprints(private_file(proof.get('storage_file', ''), parent=root).read_bytes())
    audit.compare_database(expected['database'], source_public)
    audit.compare_storage_database(expected['storage_database'], source_storage)
    actual_counts = {'tables': len(source_public['tables']),
                     'public_rows': sum(row['rows'] for row in source_public['tables'].values()),
                     'storage_objects': source_storage['objects']['rows'],
                     'storage_buckets': source_storage['buckets']['rows']}
    if any(type(proof.get(key)) is not int or proof[key] != count for key, count in actual_counts.items()):
        raise GateFailure('source_proof_counts_differ')
    digests = {'source_public_sha256': audit.digest_bytes(audit.canonical(source_public)),
               'source_storage_sha256': audit.digest_bytes(audit.canonical(source_storage))}
    if any(proof.get(key) != value for key, value in digests.items()):
        raise GateFailure('source_proof_sha256_differs')
    return {'source_proof': str(proof_path), 'source_proof_sha256': audit.digest_bytes(proof_path.read_bytes()),
            'source_captured_at_utc': proof['captured_at_utc'], **digests}


def quarantine_gate(root):
    path = private_file(root / 'quarantine-result.json', parent=root)
    proof = audit.private_json(path)
    for key in ('immutable_content_preserved', 'sql_decisions_preserved', 'source_accounts_absent'):
        if proof.get(key) is not True:
            raise GateFailure('synthetic_quarantine_proof_required')
    for key in ('real_user_records_changed', 'training_started', 'dataset_created'):
        if proof.get(key) is not False:
            raise GateFailure('synthetic_quarantine_proof_required')
    for key, count in (('objects_marked', 4), ('actual_exporter_segmentation_classification_refusals', 8)):
        if type(proof.get(key)) is not int or proof[key] != count:
            raise GateFailure('actual_exporter_refusals_required')
    snapshot_path = private_file(root / 'quarantine.originals.private.json', parent=root)
    snapshot = audit.private_json(snapshot_path)
    if proof.get('snapshot_sha256') != audit.digest_bytes(audit.canonical(snapshot)):
        raise GateFailure('quarantine_snapshot_sha256_differs')
    state = audit.private_json(root / 'fixture.private.json')
    owner, keys = quarantine.fixture_identity(state)
    identity = snapshot.get('identity', {})
    if (identity.get('tag') != state['tag'] or identity.get('owner') != owner['id']
            or identity.get('image_sha256') != state['image_sha256']
            or len(snapshot.get('records', {})) != 4
            or not keys < snapshot['records'].keys()
            or len(snapshot.get('sql_metadata', [])) != 3):
        raise GateFailure('quarantine_fixture_identity_differs')
    return {'quarantine_proof_sha256': audit.digest_bytes(path.read_bytes()),
            'quarantine_snapshot_sha256': proof['snapshot_sha256']}


def restore_gate(root, checked_backup, receipt_path, config):
    checked_backup = backup.safe_path(Path(checked_backup))
    if (checked_backup.parent != backup.BACKUP_ROOT
            or not backup.STAMP.fullmatch(checked_backup.name)
            or not (checked_backup / 'COMPLETE').is_file()):
        raise GateFailure('verified_local_backup_required')
    if backup.verify_set(checked_backup) < 1:
        raise GateFailure('verified_local_backup_required')
    receipt_path = private_file(receipt_path)
    evidence_root = root / 'backup-restore-checks'
    if (receipt_path.name != 'result.json' or receipt_path.parent.parent != evidence_root
            or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{12}', receipt_path.parent.name)):
        raise GateFailure('actual_isolated_restore_evidence_required')
    restored = audit.private_json(receipt_path)
    marker = audit.private_json(checked_backup / 'SELFHOST_BACKUP.json')
    manifest_sha = audit.digest_bytes(private_file(checked_backup / 'SHA256SUMS').read_bytes())
    if (marker.get('source_root') != str(root)
            or restored.get('event') != 'isolated_selfhost_restore_verified'
            or restored.get('restored_backup') != str(checked_backup)
            or restored.get('restore_verified') is not True
            or restored.get('backup_manifest_sha256') != manifest_sha):
        raise GateFailure('restore_backup_identity_sha256_differs')
    if (restored.get('native_sql_restore') is not True
            or restored.get('roles_owner_acl_restored_by_pg_restore') is not True
            or restored.get('network') != 'none'
            or restored.get('production_credentials_mounted') is not False
            or restored.get('worker_started') is not False
            or restored.get('training_started') is not False
            or restored.get('db_image') != config['services']['db']['image']
            or restored.get('migration_versions') != [1, 1, 1]
            or any(type(v) is not int for v in restored.get('migration_versions', []))):
        raise GateFailure('actual_isolated_sql_restore_required')
    for key, count in (('public_tables_matched', len(marker['tables'])),
                       ('storage_objects_matched', marker['objects']),
                       ('auth_sessions_rows_matched', marker['tables'].get('auth_sessions', 0)),
                       ('quality_jobs_rows_matched', marker['tables'].get('ml_jobs', 0))):
        if type(restored.get(key)) is not int or restored[key] != count:
            raise GateFailure('restored_backup_counts_differ')
    volumes = restored.get('raw_volumes', {})
    if set(volumes) != {'storage-files.tar', 'runtime-files.tar'}:
        raise GateFailure('actual_raw_volume_restore_required')
    for row in volumes.values():
        if (row.get('ownership_restored') is not True or row.get('xattrs_restored') is not True
                or type(row.get('files_restored')) is not int or row['files_restored'] < 0
                or type(row.get('directories_restored')) is not int or row['directories_restored'] < 1):
            raise GateFailure('actual_raw_volume_metadata_restore_required')
    if volumes['storage-files.tar']['files_restored'] != marker['objects']:
        raise GateFailure('restored_storage_file_count_differs')
    return {'backup_manifest_sha256': manifest_sha,
            'restore_receipt': str(receipt_path),
            'restore_receipt_sha256': audit.digest_bytes(receipt_path.read_bytes())}


def verify(root, checked_backup, restore_receipt, source_proof, config, *, now=None):
    return {**source_gate(root, source_proof, now=now), **quarantine_gate(root),
            **restore_gate(root, checked_backup, restore_receipt, config)}
