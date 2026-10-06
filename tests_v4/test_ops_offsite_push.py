"""Isolated filesystem/RPC tests; these are not a real external destination."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import io
import tempfile
import unittest
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy-vps'))
import ops_offsite_push as sender
import ops_offsite_receiver as receiver


class ExternalPushTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.backup = self.source / '20261006T150000Z'
        self.backup.mkdir()
        self.remote = self.root / 'receiver'
        self.image = 'sha256:' + 'a' * 64
        for name in ('local-files.tar', 'arborscan.env.private'):
            (self.backup / name).write_bytes(('synthetic-private-' + name).encode())
        photo = b'synthetic-photo'
        payload = json.dumps({'image_sha256': hashlib.sha256(photo).hexdigest(),
                              'image_object': 'synthetic.jpg'}).encode()
        payload_sha = hashlib.sha256(payload).hexdigest()
        reports = [{'owner_id': 'synthetic-owner', 'payload_sha256': payload_sha}]
        members = {'objects/inputs/synthetic.jpg': photo,
                   'tables/report_versions.json': json.dumps(reports).encode(),
                   'objects/inputs/report-versions/synthetic-owner/' + payload_sha + '.json': payload}
        manifest_bytes = json.dumps({n: hashlib.sha256(raw).hexdigest() for n, raw in members.items()}).encode()
        with tarfile.open(self.backup / 'application.tar', 'w') as archive:
            for name, raw in {'manifest.json': manifest_bytes, **members}.items():
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
        (self.backup / 'containers.private.json').write_text(json.dumps([
            {'Name': name, 'Image': self.image} for name in (
                '/arborscan-api-v4', '/arborscan-quality-worker')]))
        (self.backup / 'COMPLETE').touch()
        native = self.backup / 'postgres'
        native.mkdir()
        for name in ('database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt'):
            (native / name).write_bytes(('native-fixture-' + name).encode())
        (native / 'COMPLETE').touch()
        self.checksums(native)
        self.checksums(self.backup, recursive=True)
        release = self.source / 'releases' / 'fixture'
        release.mkdir(parents=True)
        manifest = {'status': 'verified_complete', 'base_images': {'file': 'base.tar'},
                    'source': {'file': 'source.tar'},
                    'overlay': {'file': 'delta.tar', 'result_image': 'sha256:' + 'b' * 64}}
        for field in ('base_images', 'source', 'overlay'):
            raw = (field + '-archive').encode()
            (release / manifest[field]['file']).write_bytes(raw)
            manifest[field]['sha256'] = hashlib.sha256(raw).hexdigest()
        (release / 'release.json').write_text(json.dumps(manifest))
        (release / 'pinned-images.json').write_text(json.dumps({'images': {'api': self.image}}))
        (release / 'delta.tar.json').write_text(json.dumps({
            'version': 1, 'base_sha256': manifest['base_images']['sha256'],
            'delta_sha256': manifest['overlay']['sha256']}))
        (release / 'COMPLETE').touch()
        self.checksums(release)
        (self.source / 'RELEASE_INDEX.json').write_text(json.dumps({
            'version': 1, 'complete': True, 'images': {self.image: {
                'release': 'releases/fixture/release.json', 'artifact': 'base_images'}}}))
        self.package, self.files = sender.build_package(self.backup, self.source)
        self.actions = []

    def checksums(self, directory, recursive=False):
        paths = directory.rglob('*') if recursive else directory.iterdir()
        files = sorted(p for p in paths if p.is_file() and p.name not in ('SHA256SUMS', 'COMPLETE'))
        if recursive:
            files += sorted(p for p in directory.rglob('COMPLETE') if p.parent != directory)
        (directory / 'SHA256SUMS').write_text(''.join(
            receiver.sha(p) + '  ' + p.relative_to(directory).as_posix() + '\n' for p in files))

    def rpc(self, request, body=b''):
        self.actions.append(request['action'])
        return receiver.dispatch(self.remote, request, body)

    def test_full_native_runtime_transfer_readback_and_idempotency(self):
        result = sender.transfer(self.package, self.files, self.rpc)
        self.assertEqual(result['files'], len(self.package['files']))
        self.assertIn('read', self.actions)
        self.assertTrue((self.remote / 'sets' / self.package['name'] / 'COMPLETE').is_file())
        self.actions.clear()
        self.assertEqual(sender.transfer(self.package, self.files, self.rpc), result)
        self.assertNotIn('put', self.actions)
        self.assertIn('read', self.actions)

    def test_lost_write_reply_resumes_without_duplicate_or_complete(self):
        interrupted = False
        def lose_reply(request, body=b''):
            nonlocal interrupted
            result = self.rpc(request, body)
            if request['action'] == 'put' and not interrupted:
                interrupted = True
                raise ValueError('lost_reply')
            return result
        with self.assertRaisesRegex(ValueError, 'lost_reply'):
            sender.transfer(self.package, self.files, lose_reply)
        self.assertFalse((self.remote / 'sets' / self.package['name'] / 'COMPLETE').exists())
        source = {name: path.read_bytes() for name, path in self.files.items()}
        sender.transfer(self.package, self.files, self.rpc)
        self.assertEqual(source, {name: path.read_bytes() for name, path in self.files.items()})
        self.assertEqual(len(list((self.remote / 'sets').iterdir())), 1)

    def test_bad_prefix_is_discarded_only_in_staging(self):
        self.rpc({'action': 'begin', 'name': self.package['name'], 'package': self.package})
        entry = next(iter(self.package['files'].values()))
        (self.remote / 'objects' / (entry['sha256'] + '.partial')).write_bytes(b'bad')
        sender.transfer(self.package, self.files, self.rpc)
        self.assertIn('reset_partial', self.actions)

    def test_receiver_refusal_keeps_all_local_archives(self):
        before = {name: receiver.sha(path) for name, path in self.files.items()}
        def refuse(request, body=b''):
            raise ValueError('receiver_refused')
        with self.assertRaisesRegex(ValueError, 'receiver_refused'):
            sender.transfer(self.package, self.files, refuse)
        self.assertEqual(before, {name: receiver.sha(path) for name, path in self.files.items()})
        self.assertFalse(self.remote.exists())

    def test_readback_checks_content_not_remote_success(self):
        def corrupt_read(request, body=b''):
            reply = self.rpc(request, body)
            if request['action'] == 'read':
                raw = base64.b64decode(reply['data'])
                reply['data'] = base64.b64encode(b'X' + raw[1:]).decode()
            return reply
        with self.assertRaisesRegex(ValueError, 'readback_checksum_mismatch'):
            sender.transfer(self.package, self.files, corrupt_read)
        self.assertFalse((self.backup.parent / 'offsite-receipts').exists())

    def test_remote_corrupt_complete_is_not_trusted(self):
        sender.transfer(self.package, self.files, self.rpc)
        digest = next(e['sha256'] for e in self.package['files'].values() if e['bytes'])
        (self.remote / 'objects' / digest).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'remote_content_not_verified'):
            self.rpc({'action': 'inspect', 'name': self.package['name']})

    def test_package_conflict_never_replaces_completed_backup(self):
        sender.transfer(self.package, self.files, self.rpc)
        changed = json.loads(json.dumps(self.package))
        changed['files']['another-safe-file'] = {'sha256': 'b' * 64, 'bytes': 1}
        with self.assertRaisesRegex(ValueError, 'existing_package_conflict'):
            self.rpc({'action': 'begin', 'name': self.package['name'], 'package': changed})
        actual = self.rpc({'action': 'inspect', 'name': self.package['name']})['package']
        self.assertEqual(actual, self.package)

    def test_path_traversal_and_missing_native_refused(self):
        changed = json.loads(json.dumps(self.package))
        changed['files']['../escaped'] = {'sha256': 'a' * 64, 'bytes': 1}
        with self.assertRaisesRegex(ValueError, 'invalid_package'):
            self.rpc({'action': 'begin', 'name': self.package['name'], 'package': changed})
        changed = json.loads(json.dumps(self.package))
        del changed['files'][self.package['name'] + '/postgres/database.dump']
        with self.assertRaisesRegex(ValueError, 'package_not_full_native_runtime'):
            sender.transfer(changed, self.files, self.rpc)

    def test_runtime_archive_and_unmapped_image_must_be_present(self):
        (self.source / 'releases' / 'fixture' / 'base.tar').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'Damaged'):
            sender.build_package(self.backup, self.source)
    def test_old_find_dot_slash_manifest_exports_without_changing_source(self):
        manifest = self.backup / 'SHA256SUMS'
        original = ''.join(row.replace('  ', '  ./', 1) + '\n' for row in manifest.read_text().splitlines()).encode()
        manifest.write_bytes(original)
        package, files = sender.build_package(self.backup, self.source)
        result = sender.transfer(package, files, self.rpc)
        self.assertEqual(result['files'], len(package['files']))
        self.assertEqual(manifest.read_bytes(), original)
    def native_runtime_fixture(self):
        assets=[]
        for name in ('current-api.tar', 'shared-base.tar.zst', 'worker.tar'):
            path=self.source/name
            path.write_bytes(('runtime-fixture-'+name).encode())
            assets.append({'path':str(path.absolute()),'sha256':receiver.sha(path)})
        runtime={'format':1,'images':{self.image:assets}}
        manifest=self.backup/'RUNTIME_DEPENDENCIES.json'
        manifest.write_text(json.dumps(runtime))
        self.checksums(self.backup,recursive=True)
        return manifest,runtime
    def test_new_runtime_dependency_contract_recovers_without_rewriting_absolute_paths(self):
        manifest,runtime=self.native_runtime_fixture()
        original=manifest.read_bytes()
        package,files=sender.build_package(self.backup,None)
        self.assertNotIn('RELEASE_INDEX.json',package['files'])
        self.assertEqual(len(package['runtime_assets']),3)
        sender.transfer(package,files,self.rpc)
        destination=self.root/'new-contract-recovery'
        sender.readback(package,self.rpc,destination)
        self.assertEqual((destination/self.backup.name/'RUNTIME_DEPENDENCIES.json').read_bytes(),original)
        relocated=json.loads((destination/'RELOCATED_RUNTIME.json').read_text())
        self.assertEqual({a['original_path'] for a in relocated['assets']},
                         {a['path'] for a in runtime['images'][self.image]})
        for item in relocated['assets']:
            self.assertEqual(receiver.sha(Path(item['restored_path'])),item['sha256'])
    def test_new_runtime_contract_corrupt_asset_is_refused(self):
        manifest,runtime=self.native_runtime_fixture()
        Path(runtime['images'][self.image][0]['path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'runtime_archive_checksum_mismatch'):
            sender.build_package(self.backup,None)
    def test_new_runtime_contract_wrong_image_or_uncovered_manifest_is_refused(self):
        manifest,runtime=self.native_runtime_fixture()
        runtime['images']={'sha256:'+'f'*64:runtime['images'][self.image]}
        manifest.write_text(json.dumps(runtime))
        self.checksums(self.backup,recursive=True)
        with self.assertRaisesRegex(ValueError,'runtime_index_image_mismatch'):
            sender.build_package(self.backup,None)
        runtime['images']={self.image:next(iter(runtime['images'].values()))}
        manifest.write_text(json.dumps(runtime))
        self.checksums(self.backup,recursive=True)
        outer=self.backup/'SHA256SUMS'
        outer.write_text(''.join(row+'\n' for row in outer.read_text().splitlines() if 'RUNTIME_DEPENDENCIES.json' not in row))
        with self.assertRaisesRegex(ValueError,'runtime_dependencies_not_hash_covered'):
            sender.build_package(self.backup,None)

    def test_recovery_downloads_all_parts_and_checks_same_native_manifest(self):
        sender.transfer(self.package, self.files, self.rpc)
        destination = self.root / 'isolated-recovery'
        result = sender.readback(self.package, self.rpc, destination)
        self.assertEqual(result['files'], len(self.package['files']))
        self.assertTrue((destination / 'EXTERNAL_RECOVERY_VERIFIED.json').is_file())
        evidence = json.loads((destination / 'EXTERNAL_RECOVERY_VERIFIED.json').read_text())
        self.assertEqual(evidence['application_restore']['reports_linked'], 1)
        self.assertFalse(evidence['database_restore_executed'])
        for name, source in self.files.items():
            self.assertEqual(source.read_bytes(), (destination / name).read_bytes())
        with self.assertRaisesRegex(ValueError, 'must_be_new'):
            sender.readback(self.package, self.rpc, destination)

    def test_receipt_is_private_and_records_actual_runtime_content(self):
        result = sender.transfer(self.package, self.files, self.rpc)
        path = self.root / 'receipts' / (self.package['name'] + '.json')
        value = sender.receipt(path, self.package, result, {'destination_id': 'authorized-host'})
        self.assertTrue(value['actual_content_verified'])
        self.assertIn('RELEASE_INDEX.json', value['assets_sha256'])
        self.assertEqual(value['manifest_sha256'], receiver.sha(self.backup / 'SHA256SUMS'))
        if os.name != 'nt':
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_same_vps_is_refused_without_ssh_or_secrets(self):
        config = {'format': 1, 'destination_id': 'same-vps', 'host': '31.57.170.88', 'user': 'arborscan',
                  'port': 22, 'remote_root': '/srv/private', 'receiver_script': '/srv/receiver.py',
                  'identity_file': '/private/key', 'known_hosts_file': '/private/known_hosts',
                  'is_independent': True, 'always_on': True}
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'destination_not_independent'):
            sender.load_config(path)
    def test_ssh_host_key_policy_and_transport_error_redaction(self):
        config={'host':'authorized-independent.invalid','user':'backup','port':2222,
                'remote_root':'/srv/private','receiver_script':'/srv/private/receiver.py',
                'identity_file':'/private/key','known_hosts_file':'/private/known_hosts'}
        transport=sender.SSHTransport(config)
        with patch('ops_offsite_push.subprocess.run',return_value=SimpleNamespace(
                returncode=1,stdout=b'',stderr=b'synthetic-sensitive-private-provider-error')) as run:
            with self.assertRaisesRegex(ValueError,'^private_transport_failed$'):
                transport({'action':'put','name':self.package['name']},b'synthetic')
            args=run.call_args.args[0]
            self.assertIn('StrictHostKeyChecking=yes',args)
            self.assertIn('BatchMode=yes',args)
            self.assertIn('IdentitiesOnly=yes',args)
            self.assertIn('UserKnownHostsFile=/private/known_hosts',args)
            self.assertEqual(run.call_count,1)
            if os.name=='posix':
                self.assertIn('ControlMaster=auto',args)
                self.assertEqual(Path(transport.control_directory.name).stat().st_mode & 0o777,0o700)
            transport.close()

    def test_duplicate_chunk_rejected_then_probe_allows_resume(self):
        self.rpc({'action': 'begin', 'name': self.package['name'], 'package': self.package})
        name, entry = next((n, e) for n, e in self.package['files'].items() if e['bytes'] > 1)
        raw = self.files[name].read_bytes()[:1]
        request = {'action': 'put', 'name': self.package['name'], 'sha256': entry['sha256'],
                   'offset': 0, 'chunk_sha256': hashlib.sha256(raw).hexdigest()}
        self.rpc(request, raw)
        with self.assertRaisesRegex(ValueError, 'chunk_offset_conflict'):
            self.rpc(request, raw)
        sender.transfer(self.package, self.files, self.rpc)
    def test_symlink_cannot_redirect_source_or_receiver(self):
        outsider = self.root / 'outside'
        outsider.mkdir()
        symlink = self.source / 'unsafe'
        try:
            symlink.symlink_to(outsider, target_is_directory=True)
        except OSError:
            self.skipTest('Windows symlink privilege unavailable; Linux test required')
        with self.assertRaisesRegex(ValueError, 'unsafe_path'):
            receiver.private_path(self.source, 'unsafe/data')
        self.remote.symlink_to(outsider, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unsafe_root'):
            self.rpc({'action': 'begin', 'name': self.package['name'], 'package': self.package})
        self.assertEqual(list(outsider.iterdir()), [])
    @unittest.skipIf(os.name == 'nt', 'receiver lock is a Linux service primitive')
    def test_real_receiver_cli_protocol_and_advisory_lock(self):
        def cli(request, body=b''):
            command = [sys.executable, '-B', str(Path(receiver.__file__)), '--root', str(self.remote)]
            result = subprocess.run(command, input=json.dumps(request).encode() + b'\n' + body,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout)
            return json.loads(result.stdout)['result']
        result = sender.transfer(self.package, self.files, cli)
        self.assertEqual(result['files'], len(self.package['files']))
        reply = cli({'action': 'inspect', 'name': self.package['name']})
        self.assertEqual(reply['package'], self.package)
    @unittest.skipIf(os.name == 'nt', 'receiver lock is a Linux service primitive')
    def test_concurrent_real_writes_cannot_publish_duplicate_chunk(self):
        name, entry = next((n, e) for n, e in self.package['files'].items() if e['bytes'] > 1)
        self.rpc({'action': 'begin', 'name': self.package['name'], 'package': self.package})
        raw = self.files[name].read_bytes()[:1]
        request = {'action': 'put', 'name': self.package['name'], 'sha256': entry['sha256'],
                   'offset': 0, 'chunk_sha256': hashlib.sha256(raw).hexdigest()}
        command = [sys.executable, '-B', str(Path(receiver.__file__)), '--root', str(self.remote)]
        def write(_):
            return subprocess.run(command, input=json.dumps(request).encode() + b'\n' + raw,
                                  capture_output=True, timeout=10).returncode
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(write, (0, 1)))
        self.assertEqual(sorted(results), [0, 1])
        self.assertEqual((self.remote / 'objects' / (entry['sha256'] + '.partial')).read_bytes(), raw)
        sender.transfer(self.package, self.files, self.rpc)


if __name__ == '__main__':
    unittest.main()
