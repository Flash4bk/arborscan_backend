"""Explicit, resumable legacy queue indexing. Dry-run unless --apply is passed.

Use the same private bucket and server credentials as v4. Never exports images,
prints identifiers, changes blobs, or marks examples eligible for training.
"""
import argparse
from urllib.parse import quote
import requests
from . import corrections_api as api


def index_legacy(apply=False):
    api._require_private_bucket()
    store = api.WorkflowStore(api._config)
    store.ready()
    url, headers, bucket = api._config()
    offset = 0
    counts = {'scanned':0, 'legacy_candidates':0, 'indexed':0}
    while True:
        response = requests.post(url+'/storage/v1/object/list/'+quote(bucket,safe=''), headers=headers,
            json={'prefix':'v4-corrections','limit':50,'offset':offset,
                  'sortBy':{'column':'name','order':'asc'}}, timeout=30)
        if response.status_code != 200: raise RuntimeError('Cannot list correction owners')
        owners = response.json()
        if not isinstance(owners,list): raise RuntimeError('Invalid owner listing')
        for folder in owners:
            owner = api._uuid(folder.get('name'))
            page_offset = 0
            while True:
                page = api.list_corrections(offset=page_offset, owner=owner)
                for item in page['items']:
                    key = item['correction_id']
                    record = api._get(owner,key)
                    counts['scanned'] += 1
                    if record.get('schema_version') == 1 and store.get(owner,key) is None:
                        counts['legacy_candidates'] += 1
                        if apply:
                            api._ensure_metadata(store,owner,key,record)
                            counts['indexed'] += 1
                if page['next_offset'] is None: break
                page_offset = page['next_offset']
        if len(owners)<50: break
        offset += len(owners)
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    try:
        print(index_legacy(args.apply))
    except Exception:
        # Do not leak storage responses, owner names or request credentials.
        parser.exit(1, 'Legacy indexing failed; check private storage and migration readiness.\n')
