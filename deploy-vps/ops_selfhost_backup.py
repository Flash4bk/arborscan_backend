"""Private local-only self-host backup. No cloud reads, rotation or deletion.

Native pg_dump and application JSON share one exported READ ONLY snapshot.
Immutable Storage bytes are read from the actual named volume. If selected
objects/buckets change while copying, the attempt remains incomplete.
SELFHOST_BACKUP.json requires an explicit self-host restore/reader branch;
legacy cloud backups and their fourteen retained sets are never rewritten.
"""
import argparse
import base64
import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading

from ops_backup_policy import locked, atomic_json, write_manifest
from ops_verify_restore import restore
from ops_runtime_contract import selfhost_inventory, transition_protection

SERVICES = {'db', 'rest', 'storage', 'gateway', 'v3', 'api', 'worker'}
PRIVATE_HOME = Path('/home/arborscan')
BACKUP_ROOT = PRIVATE_HOME / 'ops-backups'
SHA = re.compile(r'[a-f0-9]{64}')
IMAGE = re.compile(r'sha256:[a-f0-9]{64}')
STAMP = re.compile(r'\d{8}T\d{6}Z')
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
NATIVE = ('database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt')
STORAGE_SQL = "SELECT json_build_object('objects', (SELECT coalesce(json_agg(o), '[]'::json) FROM storage.objects o), 'buckets', (SELECT coalesce(json_agg(b), '[]'::json) FROM storage.buckets b));"


class BackupError(Exception):
    """Safe operational category, never source content or command stderr."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise BackupError('unsafe_private_path')
    return path


def member_name(value):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise BackupError('unsafe_object_name')
    name = PurePosixPath(value)
    if name.is_absolute() or '..' in name.parts or any(p in ('', '.') for p in value.split('/')) or '\\' in value or ':' in value:
        raise BackupError('unsafe_object_name')
    return value


def normalized_inventory(value):
    if not isinstance(value, dict) or set(value) != {'objects', 'buckets'}:
        raise BackupError('invalid_storage_inventory')
    result = {}
    for group in ('objects', 'buckets'):
        if not isinstance(value[group], list) or any(not isinstance(row, dict) for row in value[group]):
            raise BackupError('invalid_storage_inventory')
        result[group] = sorted(value[group], key=lambda row: canonical(row))
    return result


def physical_objects(inventory, tenant, layout, global_bucket):
    member_name(tenant)
    member_name(global_bucket)
    if '/' in tenant or '/' in global_bucket:
        raise BackupError('invalid_storage_tenant')
    buckets = {member_name(row['id']) for row in inventory['buckets']}
    objects = []
    seen = set()
    for row in inventory['objects']:
        bucket, name = member_name(row['bucket_id']), member_name(row['name'])
        if '/' in bucket or bucket not in buckets or (bucket, name) in seen:
            raise BackupError('invalid_storage_object')
        seen.add((bucket, name))
        size = (row.get('metadata') or {}).get('size')
        if type(size) is not int or size < 0:
            raise BackupError('unconfirmed_object_size')
        base = global_bucket + '/' + tenant + '/' + bucket + '/' + name
        version = row.get('version')
        if version is not None:
            version = member_name(version)
            if '/' in version:
                raise BackupError('invalid_storage_generation')
        if layout == 'version-directory':
            physical = base + ('/' + version if version is not None else '')
        elif layout == 'version-suffix':
            physical = base + ('__' + version if version is not None else '')
        elif layout == 'unversioned':
            physical = base
        else:
            raise BackupError('unsupported_storage_layout')
        objects.append({'physical': physical, 'logical': bucket + '/' + name, 'size': size})
    return sorted(objects, key=lambda item: item['logical'])


# This helper is run with no network and read-only volume/container mounts.
# It emits only a private tar stream, never source paths or bodies to logs.
FILE_HELPER = r'''
import base64,hashlib,io,json,os,pathlib,stat,sys,tarfile
root=pathlib.Path('/source');spec=json.load(sys.stdin);items=spec.get('objects')
if items is None:
 items=[{'physical':p.relative_to(root).as_posix()} for p in sorted(root.rglob('*')) if not p.is_dir()]
hashes={};states={};seen=set();files={};directories={};directory_states={}
def require(value):
 if not value:raise RuntimeError('Private snapshot incomplete')
def attrs(path):
 if not hasattr(os,'listxattr'):return {}
 return {n:base64.b64encode(os.getxattr(path,n,follow_symlinks=False)).decode('ascii') for n in sorted(os.listxattr(path,follow_symlinks=False))}
def metadata(path,s):
 return {'uid':s.st_uid,'gid':s.st_gid,'mode':stat.S_IMODE(s.st_mode)&0o777,'mtime_ns':s.st_mtime_ns,'xattrs':attrs(path)}
def checked(name):
 path=root.joinpath(*pathlib.PurePosixPath(name).parts)
 require(name and not pathlib.PurePosixPath(name).is_absolute() and '..' not in pathlib.PurePosixPath(name).parts)
 require('\\' not in name and ':' not in name and '.' not in name.split('/') and name not in ('manifest.json','filesystem-metadata.json'))
 require(not path.is_symlink() and all(not p.is_symlink() for p in path.parents))
 state=path.stat();require(stat.S_ISREG(state.st_mode))
 return path,state
def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def parents(path):
 parent=path.parent
 while parent.is_relative_to(root):
  name=parent.relative_to(root).as_posix();state=parent.stat();meta=metadata(parent,state)
  require(stat.S_ISDIR(state.st_mode) and not parent.is_symlink())
  if name in directories:require(directories[name]==meta and directory_states[name]==identity(state))
  else:directories[name]=meta;directory_states[name]=identity(state)
  if parent==root:break
  parent=parent.parent
def add_json(tar,name,value):
 raw=json.dumps(value,sort_keys=True).encode();info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 return hashlib.sha256(raw).hexdigest()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 parents(root/'__snapshot_root_metadata__')
 if spec.get('objects') is None:
  for directory in root.rglob('*'):
   if directory.is_dir():
    require(not directory.is_symlink());parents(directory/'__snapshot_directory_metadata__')
 for item in items:
  name=item['physical'];require(name not in seen);seen.add(name);path,before=checked(name)
  if 'size' in item:require(before.st_size==item['size'])
  parents(path);meta=metadata(path,before)
  with path.open('rb') as source:
   require(identity(os.fstat(source.fileno()))==identity(before))
   digest=hashlib.sha256()
   class Reader:
    def read(self,count):
     raw=source.read(count);digest.update(raw);return raw
   info=tarfile.TarInfo(name);info.mode=meta['mode'];info.uid=meta['uid'];info.gid=meta['gid'];info.mtime=before.st_mtime;info.size=before.st_size
   for attribute,encoded in meta['xattrs'].items():
    try:info.pax_headers['SCHILY.xattr.'+attribute]=base64.b64decode(encoded).decode('utf-8')
    except UnicodeDecodeError:pass
   tar.addfile(info,Reader())
   require(identity(os.fstat(source.fileno()))==identity(before))
  require(identity(path.stat())==identity(before) and metadata(path,path.stat())==meta)
  files[name]=meta;states[name]=identity(before);hashes[name]=digest.hexdigest()
 for name,before in states.items():
  path,state=checked(name);require(identity(state)==before and metadata(path,state)==files[name])
 for name,before in directory_states.items():
  path=root if name=='.' else root.joinpath(*pathlib.PurePosixPath(name).parts)
  state=path.stat();require(identity(state)==before and metadata(path,state)==directories[name])
 if spec.get('objects') is None:
  require(seen=={p.relative_to(root).as_posix() for p in root.rglob('*') if not p.is_dir()})
 hashes['filesystem-metadata.json']=add_json(tar,'filesystem-metadata.json',{'format':1,'files':files,'directories':directories})
 add_json(tar,'manifest.json',hashes)
'''


def verify_file_tar(path):
    """Read every actual member and hash, reject duplicate/unsafe/missing files."""
    with tarfile.open(path) as archive:
        record = archive.extractfile('manifest.json')
        if record is None:
            raise BackupError('missing_file_manifest')
        expected = json.load(record)
        if not isinstance(expected, dict):
            raise BackupError('invalid_file_manifest')
        seen = set()
        for item in archive:
            member_name(item.name)
            if not item.isfile() or item.name in seen:
                raise BackupError('unsafe_file_archive')
            seen.add(item.name)
            if item.name == 'manifest.json':
                continue
            with archive.extractfile(item) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if expected.get(item.name) != actual:
                raise BackupError('file_archive_checksum_mismatch')
        if set(expected) != seen - {'manifest.json'}:
            raise BackupError('file_archive_coverage_mismatch')
    return expected


def verify_filesystem_metadata(path, expected):
    """Validate SHA-covered permissions/xattrs; a restore must apply them explicitly."""
    if 'filesystem-metadata.json' not in expected:
        raise BackupError('missing_filesystem_metadata')
    with tarfile.open(path) as archive:
        value = json.load(archive.extractfile('filesystem-metadata.json'))
    if not isinstance(value, dict) or value.get('format') != 1:
        raise BackupError('invalid_filesystem_metadata')
    files, directories = value.get('files'), value.get('directories')
    if not isinstance(files, dict) or not isinstance(directories, dict):
        raise BackupError('invalid_filesystem_metadata')
    if set(files) != set(expected) - {'filesystem-metadata.json'}:
        raise BackupError('filesystem_metadata_coverage_mismatch')
    parents = {'.'} if files else set()
    for name in files:
        current = PurePosixPath(name).parent
        while current != PurePosixPath('.'):
            parents.add(current.as_posix())
            current = current.parent
    if not parents <= directories.keys():
        raise BackupError('filesystem_directory_coverage_mismatch')
    for group in (files, directories):
        for name, row in group.items():
            if group is files or name != '.':
                member_name(name)
            if not isinstance(row, dict) or set(row) != {'uid', 'gid', 'mode', 'mtime_ns', 'xattrs'}:
                raise BackupError('invalid_filesystem_metadata')
            if any(type(row[key]) is not int or row[key] < 0 for key in ('uid', 'gid', 'mode', 'mtime_ns')) or row['mode'] > 0o777:
                raise BackupError('invalid_filesystem_metadata')
            if not isinstance(row['xattrs'], dict):
                raise BackupError('invalid_filesystem_metadata')
            for attribute, encoded in row['xattrs'].items():
                if not isinstance(attribute, str) or not attribute or '\x00' in attribute or not isinstance(encoded, str):
                    raise BackupError('invalid_filesystem_metadata')
                try:
                    base64.b64decode(encoded, validate=True)
                except ValueError:
                    raise BackupError('invalid_filesystem_metadata') from None
    return value


def sync_directory(path):
    if os.name != 'posix':
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def restore_volume_archive(archive_path, target, *, apply_ownership=True):
    """Restore verified bytes/metadata into a new empty isolated directory only.

    No existing files are replaced or removed. Any failure leaves a partial
    isolated target for inspection; it must not be mounted in production.
    Ownership restoration requires POSIX and sufficient caller privileges.
    """
    archive_path, target = safe_path(archive_path), safe_path(target)
    expected = verify_file_tar(archive_path)
    metadata = verify_filesystem_metadata(archive_path, expected)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise BackupError('restore_target_must_be_empty')
    if apply_ownership and os.name != 'posix':
        raise BackupError('ownership_restore_requires_posix')
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    # All members were validated before any data is written. extractall is not
    # used, so tar permissions/link targets cannot redirect writes.
    with tarfile.open(archive_path) as archive:
        for name in sorted(metadata['files']):
            path = safe_path(target.joinpath(*PurePosixPath(name).parts))
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open('xb') as output, archive.extractfile(name) as source:
                shutil.copyfileobj(source, output)
                output.flush(); os.fsync(output.fileno())
            if digest(path) != expected[name]:
                raise BackupError('restored_volume_checksum_mismatch')
    for name in sorted(metadata['directories'], key=lambda n: (n.count('/'), n)):
        path = target if name == '.' else target.joinpath(*PurePosixPath(name).parts)
        safe_path(path).mkdir(mode=0o700, parents=True, exist_ok=True)
    def apply(path, row):
        safe_path(path)
        if apply_ownership:
            os.chown(path, row['uid'], row['gid'], follow_symlinks=False)
        path.chmod(row['mode'])
        if row['xattrs']:
            if not hasattr(os, 'setxattr'):
                raise BackupError('filesystem_xattr_restore_unavailable')
            for name, value in row['xattrs'].items():
                os.setxattr(path, name, base64.b64decode(value, validate=True), follow_symlinks=False)
        if os.name == 'posix':
            os.utime(path, ns=(row['mtime_ns'], row['mtime_ns']), follow_symlinks=False)
        else:
            os.utime(path, ns=(row['mtime_ns'], row['mtime_ns']))
    for name, row in metadata['files'].items():
        apply(target.joinpath(*PurePosixPath(name).parts), row)
    # Apply parents last; their original mode must not block child restoration.
    for name in sorted(metadata['directories'], key=lambda n: (n.count('/'), n), reverse=True):
        path = target if name == '.' else target.joinpath(*PurePosixPath(name).parts)
        apply(path, metadata['directories'][name])
    sync_directory(target)
    return {'files_restored': len(metadata['files']),
            'directories_restored': len(metadata['directories']),
            'ownership_restored': apply_ownership, 'xattrs_restored': True}


class PsqlSnapshot:
    def __init__(self, backend):
        self.backend = backend
        self.lines = queue.Queue()
        self.log = backend.log.open('ab')
        self.process = subprocess.Popen(backend.psql(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf-8')
        def consume():
            for line in self.process.stdout:
                self.lines.put(line.rstrip('\n'))
            self.lines.put(None)
        self.thread = threading.Thread(target=consume, daemon=True)
        self.thread.start()
        try:
            self.send('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;')
            self.identifier = self.query('SELECT pg_export_snapshot();', decode=False)
            if not re.fullmatch(r'[0-9A-Fa-f-]{1,100}', self.identifier):
                raise BackupError('invalid_exported_snapshot')
        except Exception:
            self.close()
            raise

    def send(self, sql):
        try:
            self.process.stdin.write(sql + '\n')
            self.process.stdin.flush()
        except (OSError, ValueError):
            raise BackupError('snapshot_connection_failed') from None

    def query(self, sql, decode=True):
        if decode:
            # PostgreSQL json (unlike jsonb) can retain pretty-printed newlines.
            # Frame the whole scalar result as one line so nested whitespace
            # cannot truncate a Storage/table snapshot in the live psql stream.
            statement = sql.rstrip().removesuffix(';')
            sql = ("SELECT replace(encode(convert_to((" + statement
                   + ")::text, 'UTF8'), 'base64'), E'\\n', '');")
        self.send(sql)
        try:
            line = self.lines.get(timeout=120)
        except queue.Empty:
            raise BackupError('snapshot_read_timeout') from None
        if line is None:
            raise BackupError('snapshot_connection_failed')
        try:
            return json.loads(base64.b64decode(line, validate=True)) if decode else line
        except ValueError:
            raise BackupError('invalid_snapshot_response') from None

    def tables(self):
        names = self.query("SELECT coalesce(json_agg(c.relname ORDER BY c.relname), '[]'::json) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p');")
        if not isinstance(names, list) or not names or any(not IDENTIFIER.fullmatch(n) for n in names):
            raise BackupError('invalid_table_inventory')
        return {name: self.query('SELECT coalesce(json_agg(t), \'[]\'::json) FROM public."' + name + '" t;') for name in names}

    def storage(self):
        return normalized_inventory(self.query(STORAGE_SQL))

    def close(self):
        try:
            if self.process.poll() is None:
                self.send('ROLLBACK;')
                self.process.stdin.close()
                self.process.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired, BackupError):
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        finally:
            self.thread.join(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
            self.log.close()


class DockerBackend:
    def __init__(self, root, stage):
        self.root, self.stage = root, stage
        self.log = stage / 'diagnostics.private.log'
        self.compose = json.loads((root / 'compose.private.json').read_text())
        if set(self.compose.get('services', {})) != SERVICES:
            raise BackupError('unexpected_selfhost_services')
        cmd = ['docker', 'compose', '-f', str(root / 'compose.private.json'), '--profile', 'manual-worker']
        ids = self.run(cmd + ['ps', '-aq']).decode().split()
        self.rows = json.loads(self.run(['docker', 'inspect', *ids])) if ids else []
        self.services = {r.get('Config', {}).get('Labels', {}).get('com.docker.compose.service'): r for r in self.rows}
        if set(self.services) != SERVICES or len(self.rows) != len(SERVICES):
            raise BackupError('actual_service_inventory_mismatch')
        for role, row in self.services.items():
            # Worker can be created but deliberately stopped; backing it up must
            # never launch copied jobs or training. API images also need not run.
            if (role in {'db', 'rest', 'storage', 'gateway'} and not row.get('State', {}).get('Running')) or not IMAGE.fullmatch(row.get('Image', '')):
                raise BackupError('selfhost_service_not_running')
        storage = self.services['storage']
        self.env = dict(value.split('=', 1) for value in storage['Config'].get('Env', []) if '=' in value)
        if self.env.get('STORAGE_BACKEND') != 'file':
            raise BackupError('not_local_storage_backend')
        self.volume = self.volume_for(storage, self.env.get('FILE_STORAGE_BACKEND_PATH'))
        self.runtime_mount = self.mount_for(self.services['worker'], '/app/model-quality')
        self.runtime_volume = self.runtime_mount['source']

    @staticmethod
    def volume_for(row, destination):
        matches = [m for m in row.get('Mounts', []) if m.get('Destination') == destination]
        if len(matches) != 1 or matches[0].get('Type') != 'volume' or not re.fullmatch(r'[A-Za-z0-9_.-]+', matches[0].get('Name', '')):
            raise BackupError('actual_named_volume_missing')
        return matches[0]['Name']

    def mount_for(self, row, destination):
        matches = [m for m in row.get('Mounts', []) if m.get('Destination') == destination]
        if len(matches) != 1:
            raise BackupError('actual_runtime_mount_missing')
        mount = matches[0]
        if mount.get('Type') == 'volume':
            return {'type': 'volume', 'source': self.volume_for(row, destination)}
        if mount.get('Type') != 'bind':
            raise BackupError('unsupported_runtime_mount')
        source = safe_path(mount['Source'])
        if not source.is_relative_to(self.root) or not source.is_dir() or ',' in str(source):
            raise BackupError('external_runtime_bind_source')
        return {'type': 'bind', 'source': str(source)}

    def run(self, command, *, output=None, input=None, stdin=None, timeout=1800):
        with self.log.open('ab') as error:
            result = subprocess.run(command, input=input, stdin=stdin, stdout=output or subprocess.PIPE,
                                    stderr=error, timeout=timeout)
        if result.returncode:
            raise BackupError('local_backup_command_failed')
        return result.stdout

    def db_tool(self, tool, *args):
        return ['docker', 'exec', '-i', self.services['db']['Id'], tool,
                '-U', 'supabase_admin', *args]

    def psql(self):
        return self.db_tool('psql', '-X', '-w', '-qAt', '-d', 'postgres', '-v', 'ON_ERROR_STOP=1')

    @contextlib.contextmanager
    def snapshot(self):
        current = PsqlSnapshot(self)
        try:
            yield current
        finally:
            current.close()

    def native(self, identifier, folder):
        folder.mkdir(mode=0o700)
        with (folder / 'database.dump').open('xb') as output:
            self.run(self.db_tool('pg_dump', '-w', '-d', 'postgres', '--format=custom',
                '--lock-wait-timeout=10s', '--snapshot=' + identifier), output=output)
        with (folder / 'roles.sql').open('xb') as output:
            self.run(self.db_tool('pg_dumpall', '-w', '--roles-only', '--no-role-passwords'), output=output)
        # pg_restore --list reads the actual file through stdin, not a fake header.
        with (folder / 'database.dump').open('rb') as source, (folder / 'archive-list.private.txt').open('xb') as output:
            self.run(['docker', 'exec', '-i', self.services['db']['Id'], 'pg_restore', '--list'],
                     stdin=source, output=output)
        metadata = json.loads(self.run(self.psql() + ['-c', "SELECT json_build_object('server_version', current_setting('server_version'), 'source', 'local_selfhost');"]))
        (folder / 'source.json').write_bytes(canonical(metadata))
        (folder / 'SHA256SUMS').write_text(''.join(digest(folder / n) + '  ' + n + '\n' for n in NATIVE))
        (folder / 'COMPLETE').touch(mode=0o600)

    def files(self, objects, filename, volume=None):
        mount = volume or {'type': 'volume', 'source': self.volume}
        if isinstance(mount, str):
            mount = {'type': 'volume', 'source': mount}
        if mount.get('type') == 'bind':
            source = safe_path(mount['source'])
            if not source.is_relative_to(self.root) or ',' in str(source):
                raise BackupError('external_runtime_bind_source')
        elif mount.get('type') != 'volume' or not re.fullmatch(r'[A-Za-z0-9_.-]+', mount.get('source', '')):
            raise BackupError('unsafe_volume_mount')
        with filename.open('xb') as output:
            self.run(['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only',
                '--user', '0:0', '--mount', 'type=' + mount['type'] + ',src=' + mount['source'] + ',dst=/source,readonly',
                '--entrypoint', 'python', self.services['api']['Image'], '-c', FILE_HELPER],
                input=canonical({'objects': objects}), output=output)

    def current_storage(self):
        return normalized_inventory(json.loads(self.run(self.psql() + ['-c', STORAGE_SQL])))

    def private_config(self, stage):
        configs = stage / 'compose'
        configs.mkdir(mode=0o700)
        shutil.copyfile(self.root / 'compose.private.json', configs / 'compose.private.json')
        (stage / 'containers.private.json').write_bytes(canonical(self.rows))
        effective = self.services['api']['Config'].get('Env', [])
        (stage / 'arborscan.env.private').write_text('\n'.join(effective) + '\n', encoding='utf-8')
        # Capture actual self-host bind sources, excluding production/cloud paths.
        sources = {self.root / 'compose.private.json'}
        for row in self.rows:
            labels = row.get('Config', {}).get('Labels') or {}
            for value in labels.get('com.docker.compose.project.config_files', '').split(','):
                if value:
                    source = safe_path(value)
                    if not source.is_relative_to(self.root) or not source.is_file():
                        raise BackupError('external_selfhost_compose_source')
                    sources.add(source)
            for mount in row.get('Mounts', []):
                if mount.get('Type') == 'bind':
                    source = safe_path(mount['Source'])
                    if not source.is_relative_to(self.root):
                        raise BackupError('external_selfhost_bind_source')
                    sources.add(source)
        hashes = {}
        files = set()
        for source in sources:
            if source.is_dir():
                for path in source.rglob('*'):
                    safe_path(path)
                    if not path.is_dir():
                        files.add(path)
            else:
                files.add(source)
        with tarfile.open(stage / 'local-files.tar', 'w') as archive:
            for file in sorted(files):
                safe_path(file)
                before = file.stat()
                name = 'selfhost-root/' + file.relative_to(self.root).as_posix()
                raw = file.read_bytes()
                if (file.stat().st_mtime_ns, file.stat().st_size) != (before.st_mtime_ns, before.st_size):
                    raise BackupError('config_or_model_changed')
                item = tarfile.TarInfo(name); item.mode = 0o600; item.size = len(raw)
                archive.addfile(item, io.BytesIO(raw)); hashes[name] = hashlib.sha256(raw).hexdigest()
            raw = canonical(hashes); item = tarfile.TarInfo('manifest.json'); item.mode = 0o600; item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
        for file in stage.rglob('*'):
            if file.is_file():
                file.chmod(0o600)


def application_archive(stage, objects, tables, inventory):
    raw_storage = stage / 'storage-files.tar'
    hashes = verify_file_tar(raw_storage)
    expected = {item['physical'] for item in objects}
    verify_filesystem_metadata(raw_storage, hashes)
    if set(hashes) != expected | {'filesystem-metadata.json'}:
        raise BackupError('storage_archive_inventory_mismatch')
    with tarfile.open(raw_storage) as source, tarfile.open(stage / 'application.tar', 'w') as target:
        manifest = {}
        def add(name, raw):
            member = tarfile.TarInfo(name); member.mode = 0o600; member.size = len(raw)
            target.addfile(member, io.BytesIO(raw)); manifest[name] = hashlib.sha256(raw).hexdigest()
        add('schema/buckets.json', canonical(inventory['buckets']))
        for table, rows in sorted(tables.items()):
            if not IDENTIFIER.fullmatch(table) or not isinstance(rows, list):
                raise BackupError('invalid_table_snapshot')
            add('tables/' + table + '.json', canonical(rows))
        for item in objects:
            member = copy.copy(source.getmember(item['physical']))
            if not member.isfile() or member.size != item['size']:
                raise BackupError('storage_object_size_changed')
            name = 'objects/' + item['logical']
            member.name = name
            # Application JSON/bytes use the existing logical export format.
            # Raw volume permissions/xattrs belong to storage-files.tar only.
            member.mode, member.uid, member.gid, member.pax_headers = 0o600, 0, 0, {}
            with source.extractfile(item['physical']) as stream:
                target.addfile(member, stream)
            manifest[name] = hashes[item['physical']]
        add('snapshot.json', canonical({'format': 1, 'source': 'local_selfhost',
            'transactional_database_backup': True, 'database_json_shares_native_snapshot': True,
            'objects': len(objects), 'tables': {name: len(rows) for name, rows in tables.items()}}))
        member = tarfile.TarInfo('manifest.json'); raw = canonical(manifest); member.mode = 0o600; member.size = len(raw)
        target.addfile(member, io.BytesIO(raw))


def runtime_dependencies(stage, rows, runtime_index):
    runtime = json.loads(safe_path(runtime_index).read_text())
    try:
        names, selected = selfhost_inventory(rows)
    except ValueError as error:
        raise BackupError(str(error)) from None
    if runtime.get('format') != 1 or not selected <= runtime.get('images', {}).keys():
        raise BackupError('runtime_archive_index_missing')
    verified = {}
    images = {}
    for image in sorted(selected):
        assets = runtime['images'][image]
        if not isinstance(assets, list) or not assets:
            raise BackupError('runtime_archive_missing')
        images[image] = []
        for asset in assets:
            path, expected = safe_path(asset['path']), asset['sha256']
            if not SHA.fullmatch(expected):
                raise BackupError('invalid_runtime_archive_digest')
            if path not in verified:
                verified[path] = digest(path)
            if verified[path] != expected:
                raise BackupError('runtime_archive_checksum_mismatch')
            images[image].append({'path': str(path), 'sha256': expected})
    value = {'format': 1, 'selfhost': True, 'services': names,
             'images': images}
    atomic_json(stage / 'RUNTIME_DEPENDENCIES.json', value)
    return len(verified)


def create_runtime_index(root, directory, existing_index, *, run=subprocess.run,
                         free=shutil.disk_usage):
    """Extend immutable image coverage locally, never pull or rewrite old index.

    Existing closure assets are reused only after actual SHA verification. Only
    missing local pinned images are exported. No application container starts.
    """
    root, directory, existing_index = map(safe_path, (root, directory, existing_index))
    if root.parent != PRIVATE_HOME or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', root.name):
        raise BackupError('unexpected_selfhost_root')
    expected_dir = PRIVATE_HOME / 'ops-runtime' / root.name
    if directory != expected_dir or existing_index == directory / 'RUNTIME_INDEX.json':
        raise BackupError('unexpected_runtime_index_target')
    os.umask(0o077)
    compose = json.loads((root / 'compose.private.json').read_bytes())
    if set(compose.get('services', {})) != SERVICES:
        raise BackupError('unexpected_selfhost_services')
    images = {name: row.get('image') for name, row in compose['services'].items()}
    if any(not isinstance(image, str) or not IMAGE.fullmatch(image) for image in images.values()):
        raise BackupError('exact_local_images_required')
    previous = json.loads(existing_index.read_bytes())
    if previous.get('format') != 1 or not isinstance(previous.get('images'), dict):
        raise BackupError('invalid_existing_runtime_index')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    safe_path(directory)
    target = directory / 'RUNTIME_INDEX.json'
    if target.exists():
        previous_result = json.loads(target.read_bytes())
        if (previous_result.get('format') != 1 or previous_result.get('selfhost') is not True or
                previous_result.get('service_images') != images):
            raise BackupError('existing_runtime_index_preserved')
        source_index = previous_result
    else:
        source_index = previous
    checked = {}
    result = {}
    exports = 0
    recovered = 0
    log_path = directory / 'diagnostics.private.log'
    def command(args, output=None):
        with log_path.open('ab') as error:
            reply = run(args, stdout=output or subprocess.PIPE, stderr=error, timeout=1800)
        if reply.returncode:
            raise BackupError('local_runtime_command_failed')
        return reply.stdout
    for image in sorted(set(images.values())):
        if command(['docker', 'image', 'inspect', image, '--format', '{{.Id}}']).decode().strip() != image:
            raise BackupError('local_pinned_image_missing')
        assets = source_index['images'].get(image)
        if assets:
            if not isinstance(assets, list):
                raise BackupError('invalid_runtime_asset_list')
            result[image] = []
            for asset in assets:
                path, expected = safe_path(asset['path']), asset['sha256']
                if not path.is_relative_to(PRIVATE_HOME) or not SHA.fullmatch(expected):
                    raise BackupError('unsafe_existing_runtime_asset')
                if path not in checked:
                    checked[path] = digest(path)
                if checked[path] != expected:
                    raise BackupError('runtime_archive_checksum_mismatch')
                result[image].append({'path': str(path), 'sha256': expected})
            continue
        if free(directory).free < 4 * 1024**3:
            raise BackupError('insufficient_runtime_space')
        archive = directory / (image.removeprefix('sha256:') + '.tar')
        if not archive.exists():
            partial = archive.with_suffix('.tar.partial')
            if partial.exists():
                verify_saved_image(partial, image)
                recovered += 1
            else:
                with partial.open('xb') as output:
                    command(['docker', 'image', 'save', image], output)
                    output.flush(); os.fsync(output.fileno())
                exports += 1
            # Read the real Docker-save manifest and referenced config, not just
            # a tar header. This is still not an isolated docker-load assertion.
            verify_saved_image(partial, image)
            os.replace(partial, archive)
        verify_saved_image(archive, image)
        result[image] = [{'path': str(archive), 'sha256': digest(archive)}]
    value = {'format': 1, 'selfhost': True, 'platform': {'os': 'linux', 'architecture': 'amd64'},
             'service_images': images, 'images': result}
    if target.exists() and json.loads(target.read_bytes()) != value:
        raise BackupError('existing_runtime_index_preserved')
    if not target.exists():
        atomic_json(target, value)
    sync_directory(directory)
    return {'event': 'selfhost_runtime_index_verified', 'services': len(images),
            'unique_images': len(result), 'images_exported': exports,
            'partial_archives_recovered': recovered,
            'cloud_requests': 0, 'old_index_modified': False}


def verify_saved_image(path, image):
    with tarfile.open(safe_path(path)) as archive:
        rows = json.load(archive.extractfile('manifest.json'))
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise BackupError('invalid_saved_image_manifest')
        config = member_name(rows[0].get('Config'))
        with archive.extractfile(config) as source:
            config_digest = hashlib.file_digest(source, 'sha256').hexdigest()
        expected = image.removeprefix('sha256:')
        if config_digest != expected:
            # Containerd-backed Docker uses an OCI index/manifest digest as Id.
            # docker save includes only locally available platform variants;
            # require the exact pinned index and our linux/amd64 byte closure.
            def descriptor(digest):
                if not IMAGE.fullmatch(digest):
                    raise BackupError('invalid_saved_image_descriptor')
                name = 'blobs/sha256/' + digest.removeprefix('sha256:')
                with archive.extractfile(name) as source:
                    raw = source.read()
                if hashlib.sha256(raw).hexdigest() != digest.removeprefix('sha256:'):
                    raise BackupError('saved_image_descriptor_checksum_mismatch')
                return json.loads(raw)
            try:
                root = descriptor(image)
            except KeyError:
                raise BackupError('saved_image_id_mismatch') from None
            if root.get('mediaType') in ('application/vnd.oci.image.index.v1+json',
                                       'application/vnd.docker.distribution.manifest.list.v2+json'):
                variants = [item for item in root.get('manifests', [])
                            if item.get('platform', {}).get('os') == 'linux'
                            and item.get('platform', {}).get('architecture') == 'amd64']
                if len(variants) != 1:
                    raise BackupError('saved_image_platform_missing')
                root = descriptor(variants[0]['digest'])
            if ((root.get('config') or {}).get('digest') != 'sha256:' + config_digest
                    or root.get('mediaType') not in ('application/vnd.oci.image.manifest.v1+json',
                                                    'application/vnd.docker.distribution.manifest.v2+json')):
                raise BackupError('saved_image_config_descriptor_mismatch')
            declared_layers = root.get('layers')
            if not isinstance(declared_layers, list) or not declared_layers:
                raise BackupError('saved_image_layers_missing')
            for layer in declared_layers:
                expected_layer = layer.get('digest', '')
                if not IMAGE.fullmatch(expected_layer):
                    raise BackupError('invalid_saved_image_layer_descriptor')
                with archive.extractfile('blobs/sha256/' + expected_layer.removeprefix('sha256:')) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_layer.removeprefix('sha256:'):
                        raise BackupError('saved_image_layer_checksum_mismatch')
        layers = rows[0].get('Layers')
        if not isinstance(layers, list) or not layers:
            raise BackupError('saved_image_layers_missing')
        for name in layers:
            if not archive.getmember(member_name(name)).isfile():
                raise BackupError('saved_image_layer_missing')


def verify_set(folder):
    seen = {}
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        member_name(name)
        if name in seen or not SHA.fullmatch(expected) or digest(safe_path(folder / name)) != expected:
            raise BackupError('backup_manifest_mismatch')
        seen[name] = expected
    required = {'application.tar', 'local-files.tar', 'storage-files.tar', 'runtime-files.tar',
                'arborscan.env.private', 'containers.private.json', 'SELFHOST_BACKUP.json',
                'RUNTIME_DEPENDENCIES.json'} | {'postgres/' + name for name in NATIVE}
    if not required <= seen.keys():
        raise BackupError('incomplete_selfhost_set')
    with (folder / 'postgres/database.dump').open('rb') as stream:
        if stream.read(5) != b'PGDMP':
            raise BackupError('invalid_native_archive')
    marker = json.loads((folder / 'SELFHOST_BACKUP.json').read_text())
    if marker.get('format') != 1 or marker.get('kind') != 'local_selfhost_backup':
        raise BackupError('invalid_selfhost_format')
    for name in ('storage-files.tar', 'runtime-files.tar'):
        verify_filesystem_metadata(folder / name, verify_file_tar(folder / name))
    verify_file_tar(folder / 'local-files.tar')
    return len(seen)


def build_stage(root, stage, runtime_index, *, layout, factory=DockerBackend):
    """Build/verify an unpublished private stage; shared policy can schedule it."""
    backend = factory(root, stage)
    with backend.snapshot() as snap:
        inventory = snap.storage()
        tables = snap.tables()
        objects = physical_objects(inventory, backend.env.get('TENANT_ID'), layout,
                                   backend.env.get('GLOBAL_S3_BUCKET'))
        backend.native(snap.identifier, stage / 'postgres')
        backend.files(objects, stage / 'storage-files.tar')
        backend.files(None, stage / 'runtime-files.tar', getattr(backend, 'runtime_mount', backend.runtime_volume))
        if canonical(inventory) != canonical(backend.current_storage()):
            raise BackupError('storage_changed_retry_required')
    application_archive(stage, objects, tables, inventory)
    result = restore(stage / 'application.tar', stage / 'restored')
    # Only a freshly created, verified extraction in our private staging
    # directory is disposable. Original/complete sets are never targeted.
    restored = stage / 'restored'
    if restored.parent != stage or restored.is_symlink() or any(p.is_symlink() for p in restored.rglob('*')):
        raise BackupError('unsafe_validation_directory')
    shutil.rmtree(restored)
    backend.private_config(stage)
    archives = runtime_dependencies(stage, backend.rows, runtime_index)
    atomic_json(stage / 'SELFHOST_BACKUP.json', {'format': 1, 'kind': 'local_selfhost_backup',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'source_root': str(root),
        'services': sorted(SERVICES), 'storage_layout': layout, 'storage_volume': backend.volume,
        'storage_tenant': backend.env.get('TENANT_ID'),
        'storage_global_bucket': backend.env.get('GLOBAL_S3_BUCKET'),
        'filesystem_restore_requires_metadata': True,
        'runtime_volume': backend.runtime_volume, 'objects': len(objects),
        'storage_inventory_sha256': hashlib.sha256(canonical(inventory)).hexdigest(),
        'tables': {name: len(rows) for name, rows in tables.items()},
        'runtime_archives_verified': archives, 'database_and_json_snapshot': 'same_exported_snapshot',
        'storage_consistency': 'selected_files_stat_sha_and_unchanged_metadata_interval',
        'archive_list_checked_by_pg_restore': True, 'cloud_requests': 0,
        'rotation_performed': False, 'legacy_restore_requires_selfhost_branch': True})
    atomic_json(stage / 'restore-result.json', result)
    write_manifest(stage)
    count = verify_set(stage)
    # Persist checked bytes before advertising durable completion.
    for file in stage.rglob('*'):
        if file.is_file():
            with file.open('r+b') as stream:
                os.fsync(stream.fileno())
    for directory in sorted((p for p in stage.rglob('*') if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        sync_directory(directory)
    sync_directory(stage)
    return {'objects': len(objects), 'verified_files': count}


def backup(root, output, runtime_index, *, layout, stamp=None, factory=DockerBackend,
           free=shutil.disk_usage):
    root, output = safe_path(root), safe_path(output)
    if root.parent != PRIVATE_HOME or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', root.name):
        raise BackupError('unexpected_selfhost_root')
    if output != BACKUP_ROOT:
        raise BackupError('unexpected_backup_root')
    safe_path(root / 'compose.private.json')
    stamp = stamp or dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not STAMP.fullmatch(stamp):
        raise BackupError('invalid_backup_name')
    os.umask(0o077)
    with locked(output):
        final = output / stamp
        if final.exists():
            raise BackupError('existing_backup_preserved')
        if free(output).free < 4 * 1024**3:
            raise BackupError('insufficient_backup_space')
        staging = output / '.selfhost-staging'
        staging.mkdir(mode=0o700, exist_ok=True)
        safe_path(staging)
        stage = staging / (stamp + '-' + str(os.getpid()))
        stage.mkdir(mode=0o700)
        stats = build_stage(root, stage, runtime_index, layout=layout, factory=factory)
        (stage / 'COMPLETE.partial').touch(mode=0o600)
        with (stage / 'COMPLETE.partial').open('r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(stage / 'COMPLETE.partial', stage / 'COMPLETE')
        sync_directory(stage)
        os.replace(stage, final)
        sync_directory(staging)
        sync_directory(output)
        return {'event': 'selfhost_backup_verified', 'name': stamp, **stats, 'cloud_requests': 0, 'rotation_performed': False,
                'database_restore_test_performed': False}


def scheduled_backup(root, output, runtime_index, *, layout, operation_id=None,
                     factory=DockerBackend, live=None, free=shutil.disk_usage):
    """Use the existing lock/idempotent publication/rotation after migration.

    Original transition manifests must be protected first. An unfinished cloud
    attempt requires an explicit private archive by the operator, not an implicit
    format conversion. This function never starts an API or a worker.
    """
    from ops_backup_policy import run_backup, read_json, live_dependencies
    root, output = safe_path(root), safe_path(output)
    if root.parent != PRIVATE_HOME or not re.fullmatch(r'selfhost-\d{8}T\d{6}Z', root.name) or output != BACKUP_ROOT:
        raise BackupError('unexpected_scheduled_selfhost_root')
    if not transition_protection(output):
        raise BackupError('original_backups_must_be_protected')
    active = read_json(output / 'active-backup.json', {})
    if active and not active.get('operation_id', '').startswith('selfhost-'):
        raise BackupError('unfinished_cloud_attempt_must_be_archived')
    operation_id = operation_id or 'selfhost-daily-' + dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    if not operation_id.startswith('selfhost-'):
        raise BackupError('selfhost_operation_prefix_required')
    def create(stage):
        build_stage(root, stage, runtime_index, layout=layout, factory=factory)
    return run_backup(output, create, runtime_index, operation_id=operation_id,
                      live=live_dependencies if live is None else live, free=free,
                      minimum_free=4 * 1024**3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-runtime-index', action='store_true')
    parser.add_argument('--scheduled', action='store_true')
    parser.add_argument('--operation-id')
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', default='/home/arborscan/ops-backups')
    parser.add_argument('--runtime-index', required=True,
                        help='Existing cloud index for --prepare-runtime-index; exact seven-image index otherwise')
    parser.add_argument('--storage-layout',
                        choices=('version-directory', 'version-suffix', 'unversioned'))
    args = parser.parse_args()
    try:
        if args.prepare_runtime_index:
            directory = PRIVATE_HOME / 'ops-runtime' / Path(args.root).name
            print(json.dumps(create_runtime_index(args.root, directory, args.runtime_index)))
            return
        if args.storage_layout is None:
            raise BackupError('confirmed_storage_layout_required')
        if args.scheduled:
            result = scheduled_backup(args.root, args.output, args.runtime_index,
                                      layout=args.storage_layout, operation_id=args.operation_id)
        else:
            result = backup(args.root, args.output, args.runtime_index, layout=args.storage_layout)
        print(json.dumps(result))
    except Exception:
        # Scheduled publication/rotation can precede a later receipt failure.
        # Do not report its effects as absent without inspecting the durable state.
        print(json.dumps({'event': 'selfhost_backup_failed',
                          'complete_published': None,
                          'prior_backups_modified': None if args.scheduled else False}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
