"""Explicit, account-private report snapshots. Never writes contour decisions.

Blob first, transactional index second. Unindexed blobs are not history entries;
retry verifies the same immutable blob. No automatic garbage deletion.
"""
import base64
import hashlib
import json
import math
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from fastapi.concurrency import run_in_threadpool

from .corrections_api import current_user, _config, _uuid, MAX_IMAGE, MAX_PIXELS
from .correction_workflow import WorkflowStore
from .reference_geometry import reference_geometry

router = APIRouter(prefix='/v4/reports', tags=['private report history'])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def validate_snapshot(raw, image):
    if len(raw.encode()) > 20_000_000:
        raise HTTPException(413, 'Report too large')
    try:
        data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(data, dict) or type(data.get('version')) is not int or data.get('version') != 1:
            raise ValueError()
        if set(data) - {'version','kind','report','reference','ar','environment','captured_at','change_source','correction_id'}:
            raise ValueError()
        if data.get('kind') not in ('v4','reference','legacy') or not isinstance(data.get('report'), dict):
            raise ValueError()
        if data.get('change_source') not in ('analysis','reference','manual','legacy_import','contour_link'):
            raise ValueError()
        if not isinstance(data.get('captured_at'), str):
            raise ValueError()
        datetime.fromisoformat(data['captured_at'].replace('Z','+00:00'))
        canonical(data)
        with Image.open(BytesIO(image)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise HTTPException(413,'Photo exceeds 25 megapixels')
            oriented = ImageOps.exif_transpose(im)
            width, height = oriented.size
        digest = hashlib.sha256(image).hexdigest()
        ref = data.get('reference')
        if ref is not None:
            if (type(ref['version']) is not int or ref['version'] not in (1,2) or ref['method'] != f"known_object_segment_v{ref['version']}" or
                ref['coordinates'] != 'normalized_oriented_image' or
                ref['width'] != width or ref['height'] != height or ref['same_plane'] is not True):
                raise ValueError()
            length = ref['length_m']
            if type(length) not in (int,float) or not math.isfinite(length) or length <= 0:
                raise ValueError()
            for key in ('reference','tree','crown','outline'):
                points = ref[key]
                if not isinstance(points,list) or not (3 <= len(points) <= 4096 if key=='outline' else len(points)==2):
                    raise ValueError()
                for p in points:
                    if set(p) != {'x','y'} or any(type(p[k]) not in (float,int) or not math.isfinite(p[k]) or not 0<=p[k]<=1 for k in ('x','y')):
                        raise ValueError()
            def vector(key):
                a,b=ref[key]
                return ((b['x']-a['x'])*width,(b['y']-a['y'])*height)
            rx,ry=vector('reference'); tx,ty=vector('tree'); cx,cy=vector('crown')
            squared=rx*rx+ry*ry
            if squared <= 0: raise ValueError()
            measured_height=abs(tx*rx+ty*ry)*length/squared
            crown=abs(cx*ry-cy*rx)*length/squared
            if not all(math.isfinite(x) and x>0 for x in (measured_height,crown)): raise ValueError()
            data['report']={**data['report'], 'height_m':measured_height,'crown_width_m':crown,
                            'dbh_m':None,'beta_kg_s':None,'method':ref['method']}
            if ref['version'] == 2:
                if ref.get('scale_origin') != 'vertical_reference_same_depth_user_confirmed': raise ValueError()
                data['report']['geometry'] = reference_geometry(ref)
        if data['kind']=='reference' and ref is None: raise ValueError()
        ar=data.get('ar')
        if ar is not None and (not isinstance(ar,dict) or ar.get('photo_sha256')!=digest or ar.get('association')!='user_confirmed_same_tree'):
            raise ValueError()
        # Preserve the supplied snapshot without claiming server-model provenance.
        report=data['report']
        report['beta']={'value_kg_s':None,'available':False,'reason':'dynamic_experiment_and_validated_forward_model_required'}
        report['beta_kg_s']=None
        metrics=report.get('measurements') or {}
        if not isinstance(metrics,dict): raise ValueError()
        for metric in metrics.values():
            if not isinstance(metric,dict): raise ValueError()
            value=metric.get('value_m')
            if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<=0): raise ValueError()
            if metric.get('standard')=='dbh_1_3m' and metric.get('measurement_height_m')!=1.3: raise ValueError()
        images=report.get('images') or {}
        if not isinstance(images,dict): raise ValueError()
        if images.get('width') is not None and (images['width']!=width or images.get('height')!=height): raise ValueError()
        for field in ('weather','soil','gps'):
            envelope=(data.get('environment') or {}).get(field)
            if envelope is not None and (not isinstance(envelope,dict) or 'value' not in envelope or not envelope.get('source') or not envelope.get('retrieved_at')):
                raise ValueError()
        return {**data,'image':{'sha256':digest,'width':width,'height':height,
                   'coordinates':'normalized_oriented_image','orientation':'exif_transpose',
                   'original_base64':base64.b64encode(image).decode()},
                   'provenance':'user_supplied_snapshot_not_independently_verified',
                   'units':'SI_metres_unless_explicitly_labelled'}
    except HTTPException:
        raise
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,RecursionError,OverflowError,OSError,UnidentifiedImageError,Image.DecompressionBombError):
        raise HTTPException(422,'Invalid report, reference coordinates or photo binding') from None


class ReportStore(WorkflowStore):
    def __init__(self):
        super().__init__(_config)

    def ready(self):
        if self.request('POST','rpc/server_history_version',json={}) != 1:
            raise HTTPException(503,'Server history migration unavailable')

    def blob_url(self,owner,digest):
        url,headers,bucket=_config()
        return f'{url}/storage/v1/object/{quote(bucket,safe="")}/report-versions/{_uuid(owner)}/{digest}.json',headers

    def read_blob(self,owner,digest):
        url,headers=self.blob_url(owner,digest)
        try:
            r=requests.get(url,headers=headers,timeout=60)
            if r.status_code!=200 or hashlib.sha256(r.content).hexdigest()!=digest:
                raise HTTPException(503,'Saved report file is unavailable; retry')
            return r.json()
        except (requests.RequestException,ValueError):
            raise HTTPException(503,'Saved report file is unavailable; retry') from None

    def write_blob(self,owner,payload):
        body=canonical(payload); digest=hashlib.sha256(body).hexdigest()
        url,headers=self.blob_url(owner,digest)
        try:
            r=requests.post(url,headers={**headers,'Content-Type':'application/json','x-upsert':'false'},data=body,timeout=60)
            if r.status_code not in (200,201,400,409):
                raise HTTPException(503,'Upload not confirmed; retry the same operation')
        except requests.RequestException:
            raise HTTPException(503,'Upload not confirmed; retry the same operation') from None
        self.read_blob(owner,digest)
        return digest

    def save(self,owner,analysis,version,parent,payload):
        self.ready()
        digest=self.write_blob(owner,payload)
        report=payload['report']; species=report.get('species')
        if isinstance(species,dict): species=species.get('display_name') or species.get('scientific_name')
        summary={'kind':payload['kind'],'species':species or 'Вид не определён',
                 'height_m':report.get('height_m') or (report.get('measurements',{}).get('height') or {}).get('value_m'),
                 'captured_at':payload['captured_at']}
        row=self.request('POST','rpc/save_report_version',json={'p_owner':owner,'p_analysis':analysis,
            'p_version':version,'p_parent':parent,'p_payload':digest,'p_image':payload['image']['sha256'],
            'p_correction':payload.get('correction_id'),'p_summary':summary})
        return {'saved':True,'persisted':True,'history_version':1,'record':row}

    def get(self,owner,version):
        rows=self.request('GET','report_versions',params={'owner_id':'eq.'+owner,'version_id':'eq.'+_uuid(version),'limit':'1'})
        if not rows: raise HTTPException(404,'Report not found')
        return {'record':rows[0],'snapshot':self.read_blob(owner,rows[0]['payload_sha256']),'persisted':True}

    def listing(self,owner,offset,analysis=None):
        params={'owner_id':'eq.'+owner,'order':'created_at.desc,version_id.desc','limit':'30','offset':str(offset)}
        if analysis: params['analysis_id']='eq.'+_uuid(analysis)
        rows=self.request('GET','report_versions',params=params)
        return {'items':rows,'next_offset':offset+30 if len(rows)==30 else None}


@router.get('/capabilities')
def capabilities(owner=Depends(current_user)):
    ReportStore().ready()
    return {'history_version':1, 'reference_versions':[1,2], 'geometry_version':2}


@router.post('')
async def save_report(analysis_id: str=Form(...), version_id: str=Form(...), snapshot: str=Form(...),
                      image: UploadFile=File(...), parent_id: str|None=Form(None),owner=Depends(current_user)):
    raw=await image.read(MAX_IMAGE+1)
    if not raw or len(raw)>MAX_IMAGE: raise HTTPException(413,'Invalid photo size')
    payload=await run_in_threadpool(validate_snapshot,snapshot,raw)
    return await run_in_threadpool(ReportStore().save,owner,_uuid(analysis_id),_uuid(version_id),
                                  _uuid(parent_id) if parent_id else None,payload)


@router.get('')
def list_reports(offset: int=Query(0,ge=0,le=1000000),analysis_id: str|None=None,owner=Depends(current_user)):
    return ReportStore().listing(owner,offset,analysis_id)


@router.get('/{version_id}')
def get_report(version_id: str,owner=Depends(current_user)):
    return ReportStore().get(owner,version_id)
