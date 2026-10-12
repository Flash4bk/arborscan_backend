"""Seed NEW local FileBackend bytes without changing restored Storage metadata.

Requires a PREPARED private selfhost stack and current native SQL inventory.
No Storage HTTP uploads/upserts, database writes, downloads or training.
The transient pinned Storage container has network=none and only its own new
Storage volume writable. All source/plan mounts are read-only, credentials 600.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

import ops_selfhost_stack as stack

COMPONENT = re.compile(r'[A-Za-z0-9_.-]+')
BACKEND_MODULE = '/app/dist/storage/backend/file.js'


def safe_component(value):
    if not isinstance(value, str) or not COMPONENT.fullmatch(value) or value in ('.', '..'):
        raise ValueError('Unsafe physical Storage component')
    return value


def safe_key(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid Storage object key')
    key = PurePosixPath(value)
    if (key.is_absolute() or any(part in ('.', '..') for part in value.split('/'))
            or '//' in value or value.endswith('/')):
        raise ValueError('Unsafe Storage object key')
    return value


def seed_plan(snapshot, inventory, storage_env):
    """All bytes, keys, versions and size coverage validated before any write."""
    snapshot = Path(snapshot).resolve(strict=True)
    stack.verify_snapshot(snapshot)
    expected = json.loads((snapshot / 'manifest.json').read_text())
    if not isinstance(inventory, list):
        raise ValueError('Fresh native SQL Storage inventory required')
    global_bucket = safe_component(storage_env['GLOBAL_S3_BUCKET'])
    tenant = safe_component(storage_env['TENANT_ID'])
    if storage_env['FILE_STORAGE_BACKEND_PATH'] != '/var/lib/storage' or storage_env['STORAGE_BACKEND'] != 'file':
        raise ValueError('Pinned local FileBackend configuration required')
    records, seen, paths = [], set(), set()
    for row in inventory:
        bucket = safe_component(row['bucket_id'])
        name = safe_key(row['name'])
        version = row.get('version')
        if version is not None:
            version = safe_component(version)
        relative = 'objects/' + bucket + '/' + name
        if relative in seen or relative not in expected:
            raise ValueError('Native metadata and verified objects differ')
        seen.add(relative)
        source = snapshot / relative
        metadata = row.get('metadata') or {}
        if not isinstance(metadata, dict) or type(metadata.get('size')) is not int or metadata['size'] != source.stat().st_size:
            raise ValueError('Native metadata object size differs')
        mime = metadata.get('mimetype') or 'application/octet-stream'
        cache = metadata.get('cacheControl') or 'no-cache'
        if any(not isinstance(v, str) or len(v) > 1024 or '\x00' in v or '\r' in v or '\n' in v for v in (mime, cache)):
            raise ValueError('Invalid native object header metadata')
        key = tenant + '/' + bucket + '/' + name
        target = global_bucket + '/' + key + ('/' + version if version else '')
        if target in paths:
            raise ValueError('Repeated physical Storage target')
        paths.add(target)
        records.append({'source': relative, 'global_bucket': global_bucket, 'key': key,
                        'version': version, 'physical': target, 'size': metadata['size'],
                        'sha256': expected[relative], 'mimetype': mime, 'cache_control': cache})
    existing = {name for name in expected if name.startswith('objects/')}
    if seen != existing or not seen:
        raise ValueError('Native inventory must cover every verified Storage object')
    return {'format': 1, 'backend_module': BACKEND_MODULE, 'objects': records,
            'objects_count': len(records), 'bytes': sum(row['size'] for row in records)}


# This calls the pinned filesystem adapter directly. No database client, REST or
# uploader is constructed; uploadObject here means byte write + xattrs only.
NODE_IMPORT = r'''
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const plan = JSON.parse(fs.readFileSync('/seed-plan.json', 'utf8'));
const ROOT = '/var/lib/storage';
function ensureEmpty(folder) {
  for (const name of fs.readdirSync(folder)) {
    const target = path.join(folder, name), stat = fs.lstatSync(target);
    if (stat.isSymbolicLink() || !stat.isDirectory()) throw Error('nonempty target');
    ensureEmpty(target);
  }
}
async function digest(file) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
(async () => {
  ensureEmpty(ROOT);
  const { FileBackend } = require(plan.backend_module);
  const adapter = new FileBackend();
  // Confirm the exact pinned path contract before writing any byte.
  if (adapter.filePath !== ROOT) throw Error('file backend root differs');
  for (const item of plan.objects) {
    const source = path.resolve('/seed', item.source);
    if (!source.startsWith('/seed/objects/') || fs.lstatSync(source).isSymbolicLink()
        || fs.statSync(source).size !== item.size || await digest(source) !== item.sha256)
      throw Error('source mismatch');
    const actual = await adapter.privateAssetUrl(item.global_bucket, item.key, item.version);
    if (actual !== 'local:///' + ROOT + '/' + item.physical) throw Error('physical path contract differs');
  }
  let count = 0, bytes = 0;
  for (const item of plan.objects) {
    await adapter.uploadObject(item.global_bucket, item.key, item.version,
      fs.createReadStream(path.join('/seed', item.source)), item.mimetype, item.cache_control);
    const target = path.resolve(ROOT, item.physical);
    if (!target.startsWith(ROOT + '/') || await digest(target) !== item.sha256)
      throw Error('written bytes mismatch');
    count++; bytes += item.size;
  }
  console.log(JSON.stringify({objects_seeded:count, bytes, sha256_verified:true,
    database_rows_changed:false, metadata_upsert:false, network:'none'}));
})().catch(() => {console.error('Private FileBackend seed failed; new volume retained');process.exit(1);});
'''


def identical_private_file(path, payload):
    """Retry may reuse the same private plan, never replace changed input."""
    if path.exists():
        stack.checked_private_file(path)
        if path.read_text() != payload:
            raise ValueError('Existing private seed plan differs; retained unchanged')
    else:
        stack.private_write(path, payload)


def seed(args):
    os.umask(0o077)
    root = stack.checked_resume_root(args.root)
    if not (root / 'PREPARED').is_file():
        raise ValueError('Verified private infrastructure required')
    stack.checked_private_file(root / 'compose.private.json')
    config = json.loads((root / 'compose.private.json').read_text())
    manifest = json.loads((root / 'rootmanifest.safe.json').read_text())
    if config['name'] != root.name.lower() or manifest['root'] != str(root):
        raise ValueError('Private project identity differs')
    snapshot = args.storage_snapshot.resolve(strict=True)
    metadata = args.source_storage_metadata.resolve(strict=True)
    if (str(snapshot) != manifest['storage_snapshot'] or not metadata.is_relative_to(stack.HOME)
            or metadata.is_symlink()):
        raise ValueError('Verified private source paths required')
    stack.checked_private_file(metadata)
    image = config['services']['storage']['image']
    if image != manifest['images']['storage'] or not stack.IMAGE_ID.fullmatch(image):
        raise ValueError('Pinned private Storage image required')
    runner = stack.Runner(root)
    stack.checked_infrastructure(runner, config)
    volume = config['name'] + '_storage'
    inspected = json.loads(runner.run(['docker', 'volume', 'inspect', volume]).stdout)
    if (len(inspected) != 1 or inspected[0]['Name'] != volume
            or (inspected[0].get('Labels') or {}).get('com.docker.compose.project') != config['name']):
        raise ValueError('New private Storage volume identity differs')
    plan = seed_plan(snapshot, json.loads(metadata.read_text()), config['services']['storage']['environment'])
    if args.backend_module != BACKEND_MODULE:
        # The root coordinator must verify this path from the actual pinned
        # container before changing the supported module contract in code.
        raise ValueError('Pinned backend module differs')
    plan_path = root / 'seed.plan.private.json'
    env_path = root / 'seed.env.private'
    identical_private_file(plan_path, json.dumps(plan))
    env = config['services']['storage']['environment']
    if any(not stack.ENV_KEY.fullmatch(key) or not isinstance(value, str)
           or '\n' in value or '\r' in value or '\x00' in value for key, value in env.items()):
        raise ValueError('Invalid private adapter environment')
    identical_private_file(env_path, ''.join(key + '=' + value + '\n' for key, value in env.items()))
    command = ['docker', 'run', '--rm', '--pull=never', '--network', 'none', '--user', 'root',
               '--security-opt', 'no-new-privileges:true', '--cap-drop', 'ALL',
               # Only the transient importer may read mode-600 host-owned
               # snapshot/plan bind mounts. Source mounts stay read-only; no
               # host root, production volume or network is available here.
               '--cap-add', 'DAC_OVERRIDE',
               '--memory', '512m', '--cpus', '1', '--pids-limit', '128',
               '--env-file', str(env_path), '-v', volume + ':/var/lib/storage',
               '-v', str(snapshot) + ':/seed:ro', '-v', str(plan_path) + ':/seed-plan.json:ro',
               '--entrypoint', 'node', image, '--eval', NODE_IMPORT]
    result = json.loads(runner.run(command, timeout=1800).stdout)
    if result != {'objects_seeded': plan['objects_count'], 'bytes': plan['bytes'],
                  'sha256_verified': True, 'database_rows_changed': False,
                  'metadata_upsert': False, 'network': 'none'}:
        raise RuntimeError('FileBackend seed receipt differs')
    stack.private_write(root / 'storage-seed.safe.json', json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--storage-snapshot', required=True, type=Path)
    parser.add_argument('--source-storage-metadata', required=True, type=Path)
    parser.add_argument('--backend-module', default=BACKEND_MODULE)
    args = parser.parse_args()
    try:
        print(json.dumps(seed(args)))
    except Exception:
        print('Private Storage seed incomplete; retained new volume; production unchanged.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
