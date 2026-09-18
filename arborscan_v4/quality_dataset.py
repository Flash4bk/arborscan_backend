"""Immutable tree-only datasets with explicit eligibility and raster fidelity.

No smoothing or component deletion. YOLO's polygon representation is accepted
only when its serialized round-trip preserves >=99.5% IoU of the WHOLE mask.
This is a representation-loss bound, not an inference-quality acceptance gate.
"""
import base64
import hashlib
import io
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageOps

EXPORTER_VERSION='tree-mask-fidelity-v3'


def sha(raw): return hashlib.sha256(raw).hexdigest()


def polygon_labels(mask):
    binary=(mask>127).astype(np.uint8)
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    # Retain a single instance representation; disconnected components are joined
    # with a zero-width bridge, then loss is measured after serialization.
    exterior=[c.reshape(-1,2) for i,c in enumerate(contours) if hierarchy[0][i][3]<0]
    if not exterior: raise ValueError('empty_mask')
    if len(exterior)>512 or sum(len(c) for c in exterior)>100000:raise ValueError('polygon_complexity_limit_original_retained')
    polygon=exterior[0].tolist()
    for component in exterior[1:]:
        p=np.asarray(polygon); q=component
        # Bounded nearest-point search; no quadratic allocation on large masks.
        best=(float('inf'),0,0)
        stride=max(1,len(p)//512); sampled=p[::stride]
        for j in range(0,len(q),max(1,len(q)//512)):
            d=((sampled-q[j])**2).sum(axis=1); i=int(np.argmin(d))
            if d[i]<best[0]:best=(d[i],i*stride,j)
        _,i,j=best
        loop=np.concatenate([q[j:],q[:j+1]]).tolist()
        polygon=polygon[:i+1]+loop+[polygon[i]]+polygon[i+1:]
    h,w=mask.shape
    p=np.asarray(polygon,dtype=np.float64)
    if len(p)<3 or len(p)>100000: raise ValueError('polygon_degenerate_or_too_complex')
    # Pixel centres survive float32 decoding + int32 truncation in YOLO's
    # polygon2mask. Integer corners with decimal normalization can round down.
    line='0 '+' '.join(f'{x:.9f} {y:.9f}' for x,y in (p+.5)/[w,h])
    restored=(np.array([float(v) for v in line.split()[1:]],dtype=np.float32).reshape(-1,2)*np.array([w,h],dtype=np.float32)).astype(np.int32)
    out=np.zeros_like(binary);cv2.fillPoly(out,[restored],1)
    intersection=int(np.count_nonzero(out & binary)); union=int(np.count_nonzero(out | binary))
    iou=intersection/union if union else 0
    info={'iou':iou,'lost_pixels':int(np.count_nonzero(binary & (1-out))),
          'added_pixels':int(np.count_nonzero(out & (1-binary))),
          'components':len(exterior),'holes':sum(1 for r in hierarchy[0] if r[3]>=0),
          'foreground_pixels':int(binary.sum())}
    return line, info


def sample(record, metadata, label=None, model_type='segmentation'):
    if record.get('purpose')=='export_smoke_test_only_not_model_evaluation':
        raise ValueError('synthetic_export_smoke_not_evaluation_data')
    if metadata.get('image_sha256') and metadata['image_sha256']!=record['image_sha256']:
        raise ValueError('revision_original_mismatch')
    if model_type=='segmentation':
        if metadata.get('status')!='accepted' or not any(d.get('action')=='accepted' and d.get('actor_id') and d.get('at') for d in metadata.get('decisions',[])):
            raise ValueError('accepted_revision_required')
    elif model_type=='classification':
        if not label or not label.get('confirmed') or label.get('rank')!='species':
            raise ValueError('separate_confirmed_species_label_required')
        if not label.get('authority') or not label.get('taxon_id'):
            raise ValueError('confirmed_taxon_identifier_required')
        label={**label,'class_key':label['authority']+':'+label['taxon_id']}
    else:raise ValueError('unknown_model_type')
    raw=base64.b64decode(record['original_image_base64'],validate=True)
    if sha(raw)!=record['image_sha256']:raise ValueError('original_checksum_mismatch')
    with Image.open(io.BytesIO(raw)) as im:
        if im.width*im.height>25000000:raise ValueError('image_too_large')
        rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    out=io.BytesIO();Image.fromarray(rgb).save(out,'PNG');oriented=out.getvalue()
    pixel_sha=sha(str(rgb.shape).encode()+rgb.tobytes())
    small=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(9,8),interpolation=cv2.INTER_AREA)
    dhash=''.join('1' if x else '0' for x in (small[:,1:]>small[:,:-1]).ravel())
    result={'image':oriented,'original':raw,'image_sha256':sha(raw),'pixels_sha256':pixel_sha,
       'near_duplicate_signature':dhash,'width':rgb.shape[1],'height':rgb.shape[0],
       'correction_id':metadata['correction_id'],'owner_id':metadata['owner_id'],
       'analysis_id':metadata['analysis_id'],'decision':metadata.get('decisions'),
       'group_id':(label or {}).get('group_id') or record.get('tree_series_id'),
       'label':label}
    if model_type=='segmentation':
        mask_raw=base64.b64decode(record['mask_png_base64'],validate=True)
        if sha(mask_raw)!=record['mask_sha256']:raise ValueError('mask_checksum_mismatch')
        mask=cv2.imdecode(np.frombuffer(mask_raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=rgb.shape[:2]:raise ValueError('oriented_dimensions_mismatch')
        line,fidelity=polygon_labels(mask)
        result.update(mask=mask_raw,mask_sha256=sha(mask_raw),yolo_label=line,fidelity=fidelity)
        if fidelity['iou']<0.995:raise ValueError('polygon_loss:'+json.dumps(fidelity,sort_keys=True))
    return result


def split_samples(samples,seed=20260917):
    # Union byte/pixel duplicates and known series before the split. One selected
    # revision per exact pixel image, with every excluded revision recorded.
    parent=list(range(len(samples)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    keys={}
    for i,s in enumerate(samples):
        for k in (('bytes',s['image_sha256']),('pixels',s['pixels_sha256']),
                  ('series',s['group_id']) if s.get('group_id') else ('row',i)):
            if k in keys:parent[find(i)]=find(keys[k])
            else:keys[k]=i
    roots=sorted({find(i) for i in range(len(samples))});random.Random(seed).shuffle(roots)
    holdout=max(1,len(roots)//5)
    groups={r:('test' if n<holdout else 'val' if n<2*holdout else 'train') for n,r in enumerate(roots)}
    seen=set();included=[];excluded=[]
    for i,s in enumerate(samples):
        if s['pixels_sha256'] in seen:
            excluded.append({'correction_id':s['correction_id'],'reason':'exact_duplicate_or_other_revision'});continue
        seen.add(s['pixels_sha256']);included.append({**s,'split':groups[find(i)],'split_group':find(i)})
    # Similar-image flags are advisory: never silently discard them.
    near=[]
    for i,a in enumerate(included):
        for b in included[i+1:]:
            distance=sum(x!=y for x,y in zip(a['near_duplicate_signature'],b['near_duplicate_signature']))
            if distance<=4:near.append({'a':a['correction_id'],'b':b['correction_id'],'distance':distance,
                                        'cross_split':a['split']!=b['split']})
    return included,excluded,near
