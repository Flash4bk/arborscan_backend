"""Prepare a source overlay on a recorded, immutable local runtime image.

No daemon or network is contacted unless --build is supplied. Even then this
only creates/runs a local candidate; it never replaces a container or deploys.
Version metadata is not a wheel hash lock or a security-advisory clearance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


IMAGE = re.compile(r'sha256:[a-f0-9]{64}')
COMMIT = re.compile(r'[a-f0-9]{40}')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
VERSION = re.compile(r'[A-Za-z0-9][A-Za-z0-9.+!_-]*')


def validate_manifest(value):
    if not isinstance(value, dict) or value.get('format') != 1 or set(value.get('runtimes', {})) != {'v3', 'v4'}:
        raise ValueError('invalid_runtime_manifest')
    for row in value['runtimes'].values():
        if not isinstance(row, dict) or not IMAGE.fullmatch(row.get('image_id', '')):
            raise ValueError('invalid_runtime_image')
        packages = row.get('packages')
        if (not re.fullmatch(r'3\.11\.\d+', row.get('python', '')) or row.get('platform') != 'Linux' or
                row.get('machine') != 'x86_64' or not isinstance(packages, dict) or not packages or
                any(not NAME.fullmatch(k) or not isinstance(v, str) or not VERSION.fullmatch(v)
                    for k, v in packages.items())):
            raise ValueError('invalid_runtime_metadata')
    return value


def verify_inventory(expected, observed):
    differences = []
    if observed.get('python') != expected['python']:
        differences.append('python_version_mismatch')
    packages = observed.get('packages') or {}
    if packages != expected['packages']:
        differences.append('distribution_inventory_mismatch')
    return differences


def required_sources(root, role):
    root = Path(root)
    if role not in ('v3', 'v4'):
        raise ValueError('invalid_runtime_role')
    prefixes = ['arborscan_v4'] + (['services'] if role == 'v3' else ['arborscan_science'])
    files = ['config.py'] + (['server.py', 'auth_google_identity.py', 'tree_dynamics.py'] if role == 'v3' else [])
    for name in prefixes:
        files.extend(path.relative_to(root).as_posix() for path in (root / name).rglob('*.py')
                     if '__pycache__' not in path.parts)
    mandatory = ['arborscan_v4/safe_logging.py', 'arborscan_v4/corrections_api.py',
                 'arborscan_v4/legacy_feedback.py']
    if role == 'v4':
        mandatory.extend(['arborscan_v4/science_api.py', 'arborscan_science/section.py'])
    files = sorted(set(files))
    if (any(item not in files for item in mandatory) or
            any(not re.fullmatch(r'[A-Za-z0-9_./-]+\.py', item) or '..' in Path(item).parts for item in files) or
            any(not (root / item).is_file() or (root / item).is_symlink() for item in files)):
        raise ValueError('incomplete_or_unsafe_source_closure')
    return files


def overlay_dockerfile(role, runtime, commit, files):
    if not IMAGE.fullmatch(runtime.get('image_id', '')) or not COMMIT.fullmatch(commit):
        raise ValueError('immutable_image_and_commit_required')
    lines = [f'FROM {runtime["image_id"]}',
             f'LABEL org.opencontainers.image.revision="{commit}"',
             f'LABEL org.arborscan.runtime.image="{runtime["image_id"]}"',
             f'LABEL org.arborscan.runtime.role="{role}"']
    for file in files:
        if not re.fullmatch(r'[A-Za-z0-9_./-]+\.py', file) or Path(file).anchor or '..' in Path(file).parts:
            raise ValueError('unsafe_source_path')
        lines.append(f'COPY --chown=arborscan:arborscan {file} /app/{file}')
    # Keep base command, account, dependencies, models and operating-system
    # layers exactly as recorded. A separate local run checks import ABI.
    return '\n'.join(lines) + '\n'


def validate_destination(root, destination):
    root, destination = Path(root).resolve(), Path(destination).absolute()
    allowed = root / 'output'
    if (destination.exists() or destination.is_symlink() or
            not destination.resolve().is_relative_to(allowed) or
            any(p.is_symlink() for p in [destination, *destination.parents])):
        raise ValueError('new_context_under_workspace_output_required')
    return destination


INVENTORY_CODE = '''import importlib.metadata as m,json,sys
print(json.dumps({"python":sys.version.split()[0],"packages":dict(sorted((d.metadata["Name"],d.version) for d in m.distributions()))}))
'''
IMPORT_CODE = '''import cv2,numpy,PIL,torch,torchvision,ultralytics,fastapi,requests
assert torch.version.cuda is None, "cpu_runtime_required"
import arborscan_v4.legacy_feedback,arborscan_v4.safe_logging
print("runtime_imports_ok")
'''


def context_ignore(files):
    parents = {p.as_posix() for name in files for p in Path(name).parents if p != Path('.')}
    rules = ['**', '!Dockerfile']
    for parent in sorted(parents, key=lambda name: (name.count('/'), name)):
        rules.extend(['!' + parent + '/', parent + '/**'])
    return '\n'.join([*rules, *('!' + name for name in sorted(files))]) + '\n'


def verify_context(context, runtime, role):
    """Reject a changed prepared context before contacting a Docker daemon."""
    context = Path(context)
    def linked(path):
        return path.is_symlink() or bool(getattr(path, 'is_junction', lambda: False)())
    if any(linked(path) for path in [context, *context.parents]):
        raise ValueError('unsafe_candidate_context')
    proof = context / 'provenance.json'
    if linked(proof) or not proof.is_file() or proof.stat().st_size > 1_000_000:
        raise ValueError('invalid_candidate_context')
    provenance = json.loads(proof.read_text(encoding='utf-8'))
    if not isinstance(provenance, dict) or provenance.get('format') != 1:
        raise ValueError('invalid_candidate_context')
    hashes = provenance.get('source_sha256')
    if (provenance.get('role') != role or provenance.get('base_image_id') != runtime['image_id'] or
            not COMMIT.fullmatch(provenance.get('source_commit', '')) or not isinstance(hashes, dict) or not hashes):
        raise ValueError('invalid_candidate_context')
    for name, digest in hashes.items():
        if (not re.fullmatch(r'[A-Za-z0-9_./-]+\.py', name) or Path(name).anchor or '..' in Path(name).parts or
                not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest)):
            raise ValueError('invalid_candidate_context')
    allowed = {*hashes, 'Dockerfile', '.dockerignore', 'provenance.json'}
    for path in context.rglob('*'):
        if linked(path) or (not path.is_dir() and path.relative_to(context).as_posix() not in allowed):
            raise ValueError('unexpected_candidate_context_file')
    for name, digest in hashes.items():
        path = context / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('candidate_context_source_changed')
    if ((context / 'Dockerfile').read_text(encoding='utf-8') !=
            overlay_dockerfile(role, runtime, provenance['source_commit'], sorted(hashes)) or
            (context / '.dockerignore').read_text(encoding='utf-8') != context_ignore(hashes)):
        raise ValueError('candidate_context_build_definition_changed')
    return provenance


def prepare_context(root, role, commit, destination, manifest):
    root = Path(root).resolve()
    destination = validate_destination(root, destination)
    validate_manifest(manifest)
    if not COMMIT.fullmatch(commit):
        raise ValueError('full_source_commit_required')
    resolved = subprocess.run(['git', '-C', str(root), 'rev-parse', '--verify', commit + '^{commit}'],
                              check=True, capture_output=True, text=True).stdout.strip()
    if resolved != commit:
        raise ValueError('full_source_commit_required')
    # Read exact Git object bytes, never a dirty working-tree source or env.
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory)
        names = subprocess.run(['git', '-C', str(root), 'ls-tree', '-r', '--name-only', commit],
                               check=True, capture_output=True, text=True).stdout.splitlines()
        approved = {'config.py'} | ({'server.py', 'auth_google_identity.py', 'tree_dynamics.py'} if role == 'v3' else set())
        prefixes = ('arborscan_v4/', 'services/') if role == 'v3' else ('arborscan_v4/', 'arborscan_science/')
        for name in names:
            if name in approved or (name.startswith(prefixes) and name.endswith('.py')):
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                body = subprocess.run(['git', '-C', str(root), 'show', f'{commit}:{name}'],
                                      check=True, capture_output=True).stdout
                compile(body, name, 'exec')
                target.write_bytes(body)
        files = required_sources(staged, role)
        # Only publish a complete closure. Never delete/reuse an old directory.
        destination.mkdir(parents=True)
        hashes = {}
        for name in files:
            body = (staged / name).read_bytes()
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            (destination / name).write_bytes(body)
            hashes[name] = hashlib.sha256(body).hexdigest()
    runtime = manifest['runtimes'][role]
    (destination / 'Dockerfile').write_text(overlay_dockerfile(role, runtime, commit, files), encoding='utf-8')
    (destination / '.dockerignore').write_text(context_ignore(files), encoding='utf-8')
    provenance = {'format': 1, 'role': role, 'source_commit': commit, 'base_image_id': runtime['image_id'],
                  'source_sha256': hashes, 'built': False, 'deployed': False}
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return provenance


def build_local(context, runtime, role):
    provenance = verify_context(context, runtime, role)
    image = runtime['image_id']
    observed = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                              check=True, capture_output=True, text=True).stdout.strip()
    if observed != image:
        raise ValueError('local_base_image_id_mismatch')
    candidate = f'arborscan-{role}:local-candidate-' + provenance['source_commit'][:12] + '-' + image[7:19]
    subprocess.run(['docker', 'build', '--network=none', '--pull=false', '-t', candidate,
                    '-f', str(Path(context) / 'Dockerfile'), str(context)], check=True)
    candidate_id = subprocess.run(['docker', 'image', 'inspect', candidate, '--format', '{{.Id}}'],
                                  check=True, capture_output=True, text=True).stdout.strip()
    if not IMAGE.fullmatch(candidate_id):
        raise ValueError('candidate_image_id_unconfirmed')
    run = ['docker', 'run', '--rm', '--network=none', '--read-only', '--tmpfs', '/tmp',
           '--entrypoint', 'python', candidate_id]
    value = subprocess.run([*run, '-c', INVENTORY_CODE], check=True, capture_output=True, text=True)
    if verify_inventory(runtime, json.loads(value.stdout)):
        raise ValueError('candidate_runtime_inventory_mismatch')
    # Import success alone does not prove the image contains the reviewed bytes.
    source_check = '''import hashlib,json,pathlib
expected=json.loads(%r)
for name,digest in expected.items():
 assert hashlib.sha256((pathlib.Path('/app')/name).read_bytes()).hexdigest()==digest, 'candidate_source_changed'
for prefix in {n.split('/')[0] for n in expected if '/' in n}:
 actual={p.relative_to('/app').as_posix() for p in (pathlib.Path('/app')/prefix).rglob('*.py') if '__pycache__' not in p.parts}
 assert actual=={n for n in expected if n.startswith(prefix+'/')}, 'candidate_has_stale_source'
print('candidate_source_hashes_ok')
''' % json.dumps(provenance['source_sha256'], sort_keys=True)
    subprocess.run([*run, '-c', source_check], check=True)
    imports = IMPORT_CODE + ('\nimport arborscan_science.section\n' if role == 'v4' else '')
    subprocess.run([*run, '-c', imports], check=True)
    return candidate_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', required=True, choices=['v3', 'v4'])
    parser.add_argument('--revision', required=True, help='Full committed source SHA, not a branch name')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--build', action='store_true', help='Build and check a local Docker candidate only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = validate_manifest(json.loads((root / 'deploy-vps/runtime-lock.json').read_text()))
    result = prepare_context(root, args.role, args.revision, args.output, manifest)
    if args.build:
        result['image'] = build_local(args.output, manifest['runtimes'][args.role], args.role)
        result['built'] = True
        (args.output / 'provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
