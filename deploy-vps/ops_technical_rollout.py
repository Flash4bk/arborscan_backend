"""Pinned v4/worker rollout; explicit verified backup, private plan, no v3 changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'deploy', 'rollback'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--backup', type=Path)
    parser.add_argument('--image')
    parser.add_argument('--revision')
    args = parser.parse_args()
    os.umask(0o077)
    root = args.plan.resolve()
    if not root.is_relative_to('/home/arborscan'):
        raise ValueError('Private plan must be under the operator home')
    if args.mode == 'prepare':
        backup = args.backup.resolve()
        assert (backup/'COMPLETE').is_file() and (backup/'postgres/COMPLETE').is_file()
        subprocess.run(['sha256sum', '--quiet', '-c', 'SHA256SUMS'], cwd=backup, check=True)
        image = inspect(args.image)
        assert image['Config']['Labels']['org.opencontainers.image.revision'] == args.revision
        rows = json.loads((backup/'containers.private.json').read_text())
        old = {r['Name'].lstrip('/'): r for r in rows}
        names = ['arborscan-api-v4', 'arborscan-quality-worker']
        assert all(inspect(name)['Image'] == old[name]['Image'] for name in names)
        labels = old[names[0]]['Config']['Labels']
        files = labels['com.docker.compose.project.config_files'].split(',')
        assert sha('/etc/arborscan/arborscan.env') == sha(backup/'arborscan.env.private')
        root.mkdir(mode=0o700, exist_ok=False)
        v3 = inspect('arborscan-api')
        plan = {'backup': str(backup), 'image': image['Id'], 'revision': args.revision,
                'project': labels['com.docker.compose.project'],
                'compose': files, 'compose_sha256': {p: sha(p) for p in files},
                'env_sha256': sha('/etc/arborscan/arborscan.env'),
                'old': {name: old[name]['Image'] for name in names},
                'v3': {'image': v3['Image'], 'started': v3['State']['StartedAt']},
                'models': {str(p): sha(p) for p in Path('/opt/arborscan/models').glob('*.pt')}}
        (root/'plan.private.json').write_text(json.dumps(plan, indent=2))
        for mode, api, worker in [('deploy', image['Id'], image['Id']),
                                   ('rollback', old[names[0]]['Image'], old[names[1]]['Image'])]:
            (root/(mode+'.yml')).write_text('services:\n  api-v4:\n    image: '+api+'\n  quality-worker:\n    image: '+worker+'\n')
        print('Verified backup, exact candidate, v3/model invariants and rollback prepared')
        return
    plan = json.loads((root/'plan.private.json').read_text())
    assert all(sha(p) == value for p, value in plan['compose_sha256'].items())
    assert sha('/etc/arborscan/arborscan.env') == plan['env_sha256']
    assert all(sha(p) == value for p, value in plan['models'].items())
    v3 = inspect('arborscan-api')
    assert v3['Image'] == plan['v3']['image'] and v3['State']['StartedAt'] == plan['v3']['started']
    if args.mode == 'deploy':
        assert all(inspect(name)['Image'] == value for name, value in plan['old'].items())
        subprocess.run(['docker', 'exec', 'arborscan-api-v4', 'python', '-c',
            'from arborscan_v4.model_quality_api import QualityStore; '
            'assert not QualityStore().rows("ml_jobs",state="in.(queued,running,cancel_requested)",select="id"), "Active jobs; postpone rollout"'], check=True)
    else:
        assert all(inspect(name)['Image'] in (value, plan['image']) for name, value in plan['old'].items())
    cmd = ['docker', 'compose', '-p', plan['project']]
    for file in plan['compose']:
        cmd += ['-f', file]
    cmd += ['-f', str(root/(args.mode+'.yml')), 'up', '-d', '--no-build', '--pull', 'never', '--no-deps', 'api-v4', 'quality-worker']
    with (root/(args.mode+'.private.log')).open('ab') as log:
        subprocess.run(cmd, stdout=log, stderr=log, check=True)
    print(args.mode+' finished; health/auth/worker checks are still required')


if __name__ == '__main__':
    main()
