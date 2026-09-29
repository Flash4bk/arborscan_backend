"""Reconstruct exact archive bytes from matching local blocks plus SSH ranges.

This is byte reuse, not a merge of backup records. Final manifest SHA is mandatory.
"""
import gzip,hashlib,json
from pathlib import Path
BLOCK=4*1024**2


def reconstruct(root,record,filename,target,fetch):
    expected=record.get('blocks',{}).get(filename)
    if not expected:return False
    needed={row['sha256'] for row in expected};found={}
    # Completed local sets and partially downloaded bytes may supply blocks only
    # when their content hash matches the authoritative source block exactly.
    candidates=list(root.rglob(Path(filename).name))
    partial=target.with_name(target.name+'.downloading')
    if partial.exists():candidates.append(partial)
    for candidate in candidates:
        if candidate==target or '.reconstructing' in candidate.name:continue
        with candidate.open('rb') as stream:
            offset=0
            while raw:=stream.read(BLOCK):
                digest=hashlib.sha256(raw).hexdigest()
                if digest in needed:found.setdefault(digest,(candidate,offset,len(raw)))
                offset+=len(raw)
        if needed<=found.keys():break
    missing=[row for row in expected if row['sha256'] not in found]
    if sum(row['size'] for row in missing)>32*1024**2:return False
    downloaded=fetch(missing) if missing else b''
    temporary=target.with_name(target.name+'.reconstructing');target.parent.mkdir(parents=True,exist_ok=True)
    cursor=0
    with temporary.open('wb') as output:
        for row in expected:
            if row['sha256'] in found:
                path,offset,size=found[row['sha256']]
                with path.open('rb') as source:source.seek(offset);raw=source.read(size)
            else:
                raw=downloaded[cursor:cursor+row['size']];cursor+=row['size']
            if len(raw)!=row['size'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError('delta_checksum_mismatch')
            output.write(raw)
    if cursor!=len(downloaded):raise ValueError('unexpected_delta_data')
    temporary.replace(target)
    return True
