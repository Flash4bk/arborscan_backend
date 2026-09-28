"""Verify copied backup files without loading private config or printing filenames.

Run only after transfers stop. SHA256SUMS must come from the trusted SSH source.
Extra/partial extracted files are not a substitute for the listed archives.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def verify(root, postgres=False):
    root = Path(root).resolve(strict=True)
    (root / 'OFFSITE_VERIFIED.json').unlink(missing_ok=True)
    count = total = 0
    seen = set()
    for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        normalized = str(relative)
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or relative.is_absolute() or '..' in relative.parts or '\\' in name
                or ':' in name or normalized in seen):
            raise ValueError('Invalid backup manifest')
        seen.add(normalized)
        target = (root / relative).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Backup entry outside destination or not a file')
        actual = hashlib.sha256()
        with target.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                actual.update(chunk)
                total += len(chunk)
        if actual.hexdigest() != digest:
            raise ValueError('Incomplete or damaged backup file')
        count += 1
    required = ({'database.dump', 'roles.sql', 'source.json', 'archive-list.private.txt'} if postgres else
                {'application.tar', 'local-files.tar', 'arborscan.env.private', 'containers.private.json'})
    if not required.issubset({str(PurePosixPath(n)) for n in seen}):
        raise ValueError('Required backup archives/configuration missing')
    result = {'verified_files': count, 'verified_bytes': total,
              'scope': ('native PostgreSQL archive and password-free roles' if postgres else
                        'application/files snapshot; not full PostgreSQL backup')}
    (root / 'OFFSITE_VERIFIED.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    try:
        if len(sys.argv)>2 and sys.argv[2]!='--postgres':
            raise ValueError('Unknown backup kind')
        print(json.dumps(verify(sys.argv[1], postgres=len(sys.argv)>2), indent=2))
    except (ValueError, OSError):
        # Private names and content must not leak via exception messages.
        raise SystemExit('Verification failed; destination is not a complete verified backup')
