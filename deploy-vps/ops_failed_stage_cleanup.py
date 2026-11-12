"""Explicit-scope failed-stage cleanup. Default is full read-only plan.

Input is private reviewed JSON on stdin. --apply-proof-sha must equal the
separately reviewed offhost proof hash. All target paths/attempts are explicit;
preparing or running a plan does not authorize deletion. No SQL/API/worker actions.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,shutil,stat,subprocess,sys,re
from ops_failed_stage_quarantine import decode_json

from dataclasses import dataclass

@dataclass(frozen=True)
class Scope:
    root: Path
    parent: Path
    names: tuple
    proof: str
    operation_id: str
    active_attempt: int
    minimum_complete: int = 2

    def validate(self):
        if (not self.root.is_absolute() or self.parent.parent != self.root / '.staging'
                or not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z', self.parent.name)
                or not re.fullmatch(r'[a-f0-9]{64}', self.proof)
                or not self.names or len(set(self.names)) != len(self.names)
                or len(self.names) > 100 or self.minimum_complete < 2
                or any(not re.fullmatch(r'attempt-[1-9][0-9]*', name) for name in self.names)
                or 'attempt-' + str(self.active_attempt) in self.names):
            raise ValueError('invalid_explicit_cleanup_scope')


def fingerprint(s):
    return (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode))

def safe(path):
    if path.resolve()!=path or any(q.is_symlink() for q in [path,*path.parents]):
        raise ValueError('unsafe_source')
    s=path.lstat()
    if s.st_uid!=os.getuid() or not(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):
        raise ValueError('source_owner_or_type_changed')
    return s

def verify_tree(request, scope):
    scope.validate()
    if request.get('proof_sha256')!=scope.proof or request.get('source_parent')!=str(scope.parent):
        raise ValueError('wrong_reviewed_proof')
    proof_raw=base64.b64decode(request['proof_raw_base64'],validate=True)
    if hashlib.sha256(proof_raw).hexdigest()!=scope.proof:
        raise ValueError('cleanup_proof_checksum')
    proof=decode_json(proof_raw)
    layout_raw=base64.b64decode(request['layout_gzip_base64'],validate=True)
    if hashlib.sha256(layout_raw).hexdigest()!=proof['layout_sha256']:
        raise ValueError('cleanup_layout_checksum')
    layout=decode_json(layout_raw,compressed=True)
    if (proof['source_parent']!=str(scope.parent)
            or set(proof['attempts'])!={str(scope.parent/name) for name in scope.names}
            or {name:{'bytes':r['bytes'],'sha256':r['sha256']} for name,r in request['files'].items()}!=proof['source_file_sha']
            or request['directories']!=layout['directories']):
        raise ValueError('cleanup_scope_not_bound_to_quarantine')
    for row in layout['files']:
        current=request['files'][row['relative']]
        if any(current[key]!=row[key] for key in ['uid','gid','mode','mtime_ns','xattrs','sha256','bytes']):
            raise ValueError('cleanup_metadata_not_bound_to_quarantine')
    safe(scope.parent)
    active_path=scope.root/'active-backup.json';active_bytes=active_path.read_bytes();active=json.loads(active_bytes)
    if (active.get('name')!=scope.parent.name or active.get('operation_id')!=scope.operation_id
            or active.get('attempt')!=scope.active_attempt or not(scope.parent/('attempt-'+str(scope.active_attempt))).is_dir()):
        raise ValueError('active_operation_changed')
    completed={p.name for p in scope.root.iterdir() if p.is_dir() and (p/'COMPLETE').is_file()}
    if len(completed)<scope.minimum_complete:raise ValueError('protected_complete_inventory_changed')
    expected=request['files'];directories={r['relative']:r for r in request['directories']}
    seen=set();seen_dirs=set();logical=0;allocated=0;states={}
    for name in scope.names:
        root=scope.parent/name;safe(root)
        if (root/'COMPLETE').exists() or not(root/'postgres/COMPLETE').is_file():
            raise ValueError('candidate_completion_or_ownership_changed')
        for path in [root,*sorted(root.rglob('*'))]:
            before=safe(path);relative=path.relative_to(scope.parent).as_posix()
            row=directories.get(relative) if path.is_dir() else expected.get(relative)
            if row is None:raise ValueError('unrecorded_source_entry')
            attrs={k:base64.b64encode(os.getxattr(path,k,follow_symlinks=False)).decode()
                   for k in os.listxattr(path,follow_symlinks=False)}
            if (before.st_uid,before.st_gid,stat.S_IMODE(before.st_mode),before.st_mtime_ns,attrs)!=(row['uid'],row['gid'],row['mode'],row['mtime_ns'],row['xattrs']):
                raise ValueError('source_metadata_changed')
            if path.is_dir():
                seen_dirs.add(relative)
                states[str(path)]=(fingerprint(before),True)
                continue
            if before.st_nlink!=1 or fingerprint(before)!=(row['inode'],row['bytes'],row['mtime_ns'],row['ctime_ns'],row['uid'],row['gid'],row['mode']):
                raise ValueError('source_file_identity_changed')
            with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as source:
                if fingerprint(os.fstat(source.fileno()))!=fingerprint(before):raise ValueError('source_race')
                actual=hashlib.file_digest(source,'sha256').hexdigest()
                if fingerprint(os.fstat(source.fileno()))!=fingerprint(before):raise ValueError('source_race')
            if actual!=row['sha256'] or fingerprint(safe(path))!=fingerprint(before):raise ValueError('source_file_checksum_changed')
            seen.add(relative);logical+=before.st_size;allocated+=before.st_blocks*512
            states[str(path)]=(fingerprint(before),False)
    if seen!=set(expected) or seen_dirs!=set(directories):raise ValueError('source_tree_changed')
    return {'format':1,'kind':'reviewed_failed_attempt_cleanup_plan','proof_sha256':scope.proof,
            'paths':[str(scope.parent/n) for n in scope.names],'source_files_sha_verified':len(seen),
            'logical_bytes':logical,'allocated_file_bytes':allocated,'free_before':shutil.disk_usage(scope.root).free,
            'complete_sets_retained':len(completed),'active_attempt_retained':scope.active_attempt,'source_deleted':False},states,active_bytes,completed


def recheck_tree(scope,states,name):
    """Exact membership and every file/directory identity, including the root."""
    root=scope.parent/name
    if name not in scope.names or root.resolve()!=root or root.parent!=scope.parent:
        raise ValueError('wrong_delete_target')
    if (root/'COMPLETE').exists():raise ValueError('candidate_completion_changed')
    expected={Path(path):state for path,state in states.items() if Path(path).is_relative_to(root)}
    actual={root,*root.rglob('*')}
    if actual!=set(expected):raise ValueError('source_tree_membership_changed')
    for path,(identity,is_directory) in expected.items():
        current=safe(path)
        if fingerprint(current)!=identity or stat.S_ISDIR(current.st_mode)!=is_directory:
            raise ValueError('source_tree_identity_changed')
    return expected


def delete_verified_entries(scope,states):
    """Delete only recorded names, never recursively remove unknown contents.

    A new entry before apply rejects before any unlink. A concurrent new entry
    after that check is never targeted: bottom-up rmdir refuses its directory.
    A partial deletion of already-copied known entries remains recoverable.
    """
    scope.validate()
    if scope.root.resolve()!=scope.root:raise ValueError('wrong_source_root')
    for name in scope.names:recheck_tree(scope,states,name)
    for name in scope.names:
        expected=recheck_tree(scope,states,name)
        for path,(identity,is_directory) in expected.items():
            if is_directory:continue
            current=safe(path)
            if fingerprint(current)!=identity or not stat.S_ISREG(current.st_mode):
                raise ValueError('source_file_identity_changed')
            path.unlink()
        for path,(identity,is_directory) in sorted(expected.items(),key=lambda item:len(item[0].parts),reverse=True):
            if not is_directory:continue
            current=safe(path)
            # Our own unlinks change directory size/timestamps. Keep immutable
            # identity/ownership/mode checks, then rmdir protects new contents.
            now=fingerprint(current)
            if (not stat.S_ISDIR(current.st_mode)
                    or (now[0],*now[4:])!=(identity[0],*identity[4:])):
                raise ValueError('source_directory_identity_changed')
            path.rmdir()

def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-root',type=Path,required=True)
    parser.add_argument('--source-parent',type=Path,required=True)
    parser.add_argument('--attempt',action='append',required=True)
    parser.add_argument('--proof-sha256',required=True)
    parser.add_argument('--operation-id',required=True)
    parser.add_argument('--active-attempt',type=int,required=True)
    parser.add_argument('--minimum-complete',type=int,default=2)
    parser.add_argument('--service',default='arborscan-backup.service')
    parser.add_argument('--apply-proof-sha')
    args=parser.parse_args();request=json.load(sys.stdin)
    scope=Scope(args.backup_root,args.source_parent,tuple(args.attempt),args.proof_sha256,
                args.operation_id,args.active_attempt,args.minimum_complete)
    scope.validate()
    if scope.root.resolve()!=scope.root:raise ValueError('wrong_source_root')
    safe(scope.root);lock=scope.root/'backup.lock';safe(lock)
    descriptor=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'r+') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reply=subprocess.run(['systemctl','show',args.service,'-p','ActiveState','--value'],capture_output=True,timeout=15,check=True)
        if reply.stdout.strip() not in [b'failed',b'inactive']:raise ValueError('service_must_be_inactive')
        plan,states,active_bytes,completed=verify_tree(request,scope)
        if args.apply_proof_sha is not None:
            if args.apply_proof_sha!=scope.proof:
                raise ValueError('explicit_scoped_proof_required')
            if (scope.root/'active-backup.json').read_bytes()!=active_bytes:raise ValueError('active_operation_changed')
            delete_verified_entries(scope,states)
            if any((scope.parent/name).exists() for name in scope.names):raise ValueError('cleanup_incomplete')
            if ((scope.root/'active-backup.json').read_bytes()!=active_bytes
                    or not(scope.parent/('attempt-'+str(scope.active_attempt))).is_dir()):raise ValueError('protected_operation_changed')
            if not all((scope.root/name/'COMPLETE').is_file() for name in completed):raise ValueError('protected_complete_changed')
            plan.update(source_deleted=True,free_after=shutil.disk_usage(scope.root).free,
                        rollback_requires_reviewed_offhost_quarantine=True)
        print(json.dumps(plan),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        print(json.dumps({'event':'reviewed_cleanup_refused','reason':'private_validation_or_scope_changed'}),flush=True)
        raise SystemExit(1)
