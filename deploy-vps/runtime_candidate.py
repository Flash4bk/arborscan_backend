"""Prepare a source overlay on a recorded, immutable local runtime image.

No daemon or network is contacted unless --build is supplied. Even then this
only creates/runs a local candidate; it never replaces a container or deploys.
Version metadata is not a wheel hash lock or a security-advisory clearance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
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


INVENTORY_CODE = '''import importlib.metadata as m,json,pathlib,sys
private=sum(1 for p in pathlib.Path('/app').rglob('*') if p.is_file() and
 (p.name=='.env' or p.name.startswith('.env.') or p.name.endswith(('.env','.env.private'))))
print(json.dumps({"python":sys.version.split()[0],"packages":dict(sorted((d.metadata["Name"],d.version) for d in m.distributions())),"private_configuration_files":private}))
'''
IMPORT_CODE = '''import cv2,numpy,PIL,torch,torchvision,ultralytics,fastapi,requests
assert torch.version.cuda is None, "cpu_runtime_required"
assert torch.equal(torch.add(torch.tensor([1.,2.]),torch.tensor([1.,2.])),torch.tensor([2.,4.]))
assert torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]),torch.tensor([.9,.8]),.5).tolist()==[0]
assert cv2.cvtColor(numpy.zeros((1,1,3),dtype=numpy.uint8),cv2.COLOR_BGR2GRAY).shape==(1,1)
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


def docker_run(arguments, **kwargs):
    # Match every image operation to the explicitly selected local builder.
    # A remote DOCKER_HOST/CONTEXT override must not divert just half the calls.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('DOCKER_', 'BUILDX_'))}
    return subprocess.run(['docker', '--context', 'default', *arguments], env=environment, **kwargs)


def inspect_local_image(reference, *, required=True):
    reply = docker_run(['image', 'inspect', reference, '--format', '{{json .}}'],
                       capture_output=True, text=True)
    if reply.returncode:
        if required:
            raise ValueError('local_image_unavailable')
        return None
    value = json.loads(reply.stdout)
    if not isinstance(value, dict) or not IMAGE.fullmatch(value.get('Id', '')):
        raise ValueError('invalid_local_image_metadata')
    return value


def smoke_command(image, environment_keys):
    # Clear every inherited image setting, including unexpected private keys.
    # No production env, model or configuration path is mounted or imported.
    isolated = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPYCACHEPREFIX': '/tmp/as14-pycache',
                'XDG_CONFIG_HOME': '/tmp/as14-config', 'YOLO_CONFIG_DIR': '/tmp/as14-ultralytics',
                'YOLO_OFFLINE': 'true', 'YOLO_AUTOINSTALL': 'false',
                'TORCH_HOME': '/tmp/as14-torch', 'MPLCONFIGDIR': '/tmp/as14-matplotlib',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    environment = dict.fromkeys(environment_keys, '') | isolated
    args = ['run', '--rm', '--network=none', '--pull=never', '--read-only',
            '--user', 'arborscan', '--workdir', '/app', '--memory', '2g', '--cpus', '1',
            '--pids-limit', '128', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=134217728,mode=1777']
    for key, value in sorted(environment.items()):
        args.extend(['--env', key + '=' + value])
    return [*args, '--entrypoint', 'python', image, '-B']


def build_local(context, runtime, role, *, audit=None):
    provenance = verify_context(context, runtime, role)
    image = runtime['image_id']
    # `inspect` has no --format in Buildx 0.37; `ls` exposes JSON lines.
    # Docker may list the same default node twice for its context and builder.
    listing = docker_run(['buildx', 'ls', '--format', '{{json .}}'],
                         check=True, capture_output=True, text=True).stdout
    builders = []
    for line in listing.splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('Name') == 'default':
                builders.append(row)
    if len({json.dumps(row, sort_keys=True) for row in builders}) != 1:
        raise ValueError('ambiguous_or_missing_default_builder')
    builder = builders[0]
    if (builder.get('Driver') != 'docker' or len(builder.get('Nodes', [])) != 1
            or builder['Nodes'][0].get('Endpoint') != 'default'
            or builder['Nodes'][0].get('Status') != 'running'):
        raise ValueError('local_default_docker_builder_required')
    base = inspect_local_image(image)
    if base['Id'] != image:
        raise ValueError('local_base_image_id_mismatch')
    original_references = set(base.get('RepoTags') or []) | set(base.get('RepoDigests') or [])
    if not original_references:
        raise ValueError('original_base_image_reference_required')
    config = base.get('Config') or {}
    if config.get('Volumes') or config.get('User') != 'arborscan' or config.get('WorkingDir') != '/app':
        raise ValueError('unexpected_base_image_configuration')
    environment_keys = sorted({entry.partition('=')[0] for entry in config.get('Env', [])})
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) for key in environment_keys):
        raise ValueError('unexpected_base_image_environment_key')
    job = secrets.token_hex(6)
    alias = 'arborscan-runtime-base:' + role + '-' + job
    candidate = (f'arborscan-{role}:local-candidate-' + provenance['source_commit'][:12]
                 + '-' + image[7:19] + '-' + job)
    if inspect_local_image(alias, required=False) or inspect_local_image(candidate, required=False):
        raise ValueError('existing_candidate_or_base_alias_preserved')
    original = (Path(context) / 'Dockerfile').read_bytes()
    effective = ('FROM ' + alias + '\n').encode() + original.split(b'\n', 1)[1]
    proof = {'builder': 'default', 'driver': 'docker', 'base_image_id': image,
             'base_alias': alias, 'candidate_tag': candidate,
             'original_dockerfile_sha256': hashlib.sha256(original).hexdigest(),
             'effective_dockerfile_sha256': hashlib.sha256(effective).hexdigest(),
             'inherited_environment_keys': environment_keys,
             'original_base_references': sorted(original_references), 'base_alias_removed': False}
    if audit is not None:
        audit.update(proof)
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
    created = False
    try:
        docker_run(['image', 'tag', image, alias], check=True)
        created = True
        if inspect_local_image(alias)['Id'] != image:
            raise ValueError('local_base_alias_changed')
        # BuildKit cannot use `FROM sha256:<config-id>`. The default Docker
        # driver with pull=false resolves this owned local alias first. Keep
        # the committed context unchanged; only this temporary FROM differs.
        with tempfile.TemporaryDirectory(prefix='as14-runtime-recipe-') as temporary:
            dockerfile = Path(temporary) / 'Dockerfile'
            dockerfile.write_bytes(effective)
            docker_run(['buildx', 'build', '--builder', 'default', '--network=none',
                        '--pull=false', '--provenance=false', '-t', candidate,
                        '-f', str(dockerfile), str(context)], check=True)
        if inspect_local_image(alias)['Id'] != image:
            raise ValueError('local_base_alias_changed')
        verify_context(context, runtime, role)
        candidate_id = inspect_local_image(candidate)['Id']
        run = smoke_command(candidate_id, environment_keys)
        value = docker_run([*run, '-c', INVENTORY_CODE], check=True, capture_output=True, text=True)
        observed = json.loads(value.stdout)
        if observed.get('private_configuration_files') != 0:
            raise ValueError('candidate_private_configuration_files_present')
        if verify_inventory(runtime, observed):
            raise ValueError('candidate_runtime_inventory_mismatch')
        docker_run([*run, '-c', source_check], check=True)
        imports = IMPORT_CODE + ('\nimport arborscan_science.section\n' if role == 'v4' else '')
        docker_run([*run, '-c', imports], check=True)
        return candidate_id
    finally:
        if created:
            current = inspect_local_image(alias)
            if current['Id'] != image:
                raise ValueError('local_base_alias_changed_preserved')
            references = set(current.get('RepoTags') or []) | set(current.get('RepoDigests') or [])
            if not original_references <= references:
                raise ValueError('original_base_image_references_changed_preserved')
            docker_run(['image', 'rm', '--no-prune', alias], check=True)
            if audit is not None:
                audit['base_alias_removed'] = True


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
        audit = {}
        result['image'] = build_local(args.output, manifest['runtimes'][args.role], args.role, audit=audit)
        result['build_audit'] = audit
        result['built'] = True
        (args.output / 'provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
