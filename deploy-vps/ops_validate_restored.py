"""Offline verification in a network-disabled runtime, without production env."""
import base64
import hashlib
import io
import json
import pathlib
import sys
from PIL import Image

root=pathlib.Path(sys.argv[1])
rows=json.loads((root/'tables/report_versions.json').read_text())
photos=0;contours=0
for row in rows:
    paths=list((root/'objects').glob('*/report-versions/'+row['owner_id']+'/'+row['payload_sha256']+'.json'))
    assert len(paths)==1
    payload=json.loads(paths[0].read_bytes())
    raw=base64.b64decode(payload['image']['original_base64'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==row['image_sha256']
    with Image.open(io.BytesIO(raw)) as image:image.verify()
    photos+=1
for row in json.loads((root/'tables/contour_revisions.json').read_text()):
    paths=list((root/'objects').glob('*/v4-corrections/'+row['owner_id']+'/'+row['correction_id']))
    assert len(paths)==1
    payload=json.loads(paths[0].read_bytes())
    for key in ('original_image_base64','mask_png_base64'):
        raw=base64.b64decode(payload[key],validate=True)
        with Image.open(io.BytesIO(raw)) as image:image.verify()
    contours+=1
print(json.dumps({'report_photos_opened':photos,'contour_pairs_opened':contours,
                  'database_restored':False,'network':'disabled_by_docker_run'}))
