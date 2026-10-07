# Prepared only: this command was NOT executed.
python3 - <<'PY'
import fcntl, hashlib, os, subprocess, tempfile
from pathlib import Path
root = Path('/home/arborscan/ops-backups')
target = Path('/home/arborscan/ops-tools/ops_backup_policy.py')
saved = Path('/home/arborscan/ops-policy-before-20261007T065203Z/ops_backup_policy.py')
expected = 'fceb94e93aed32fdeea2e11dbe2c9a02842a798ed56266226e7436f16a105664'
current = '2aa70c5c6541ae2d76ac83f7a01c755186e24dd1408633a8c49ebac7c4a3b933'
with (root/'backup.lock').open('a') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = subprocess.check_output(['systemctl','show','arborscan-backup.service','-p','ActiveState','-p','MainPID'],text=True)
    status = dict(line.split('=',1) for line in state.splitlines() if '=' in line)
    assert status == {'MainPID':'0','ActiveState':'inactive'}
    assert not (root/'active-backup.json').exists()
    assert not any(p.is_symlink() for p in (root,root/'backup.lock',saved,saved.parent,target,target.parent))
    payload = saved.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected
    assert hashlib.sha256(target.read_bytes()).hexdigest() == current
    fd, temporary = tempfile.mkstemp(prefix='.policy-rollback-',dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as out:
            out.write(payload); out.flush(); os.fsync(out.fileno())
        os.chmod(temporary,0o600)
        os.replace(temporary,target)
        directory = os.open(target.parent,os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    print('policy rollback verified; API, DB, timer and stored copies unchanged')
PY
