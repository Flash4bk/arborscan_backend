"""Admin ML workflow. This module never trains inside an HTTP request."""
import base64
import io
import json
import os
import tarfile
import time
from pathlib import Path
from uuid import uuid4
import requests
from fastapi import APIRouter,Depends,HTTPException
from pydantic import BaseModel,Field
from .corrections_api import current_admin,_config,_uuid,_get
from .correction_workflow import WorkflowStore
from .report_history import ReportStore,canonical
from .quality_dataset import sample,split_samples,sha,EXPORTER_VERSION

router=APIRouter(prefix='/v4/model-quality',tags=['admin model quality'])


class QualityStore(WorkflowStore):
    def __init__(self):super().__init__(_config)
    def transition(self,action,id,actor=None,**data):
        return self.request('POST','rpc/ml_transition',json={'p_action':action,'p_id':id,'p_actor':actor,'p_data':data})
    def rows(self,table,**params):return self.request('GET',table,params=params)
    def one(self,table,id):
        r=self.rows(table,id='eq.'+_uuid(id),limit='1')
        if not r:raise HTTPException(404,'ML record unavailable')
        return r[0]


ARCHIVE_CHUNK_BYTES=8*1024*1024
MAX_ARCHIVE_BYTES=256*1024*1024


def _asset_json(url,headers,payload=None):
    try:
        if payload is not None:
            r=requests.post(url,headers={**headers,'Content-Type':'application/json','x-upsert':'false'},data=canonical(payload),timeout=120)
            if r.status_code not in (200,201,400,409):raise HTTPException(503,'Snapshot upload unconfirmed')
        r=requests.get(url,headers=headers,timeout=120)
        if r.status_code!=200:raise HTTPException(503,'Snapshot object unavailable')
        value=r.json()
        if not isinstance(value,dict) or (payload is not None and value!=payload):raise HTTPException(503,'Snapshot object mismatch')
        return value
    except (requests.RequestException,ValueError):raise HTTPException(503,'Snapshot storage unavailable') from None


def asset(digest,raw=None):
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid asset digest')
    u,h,b=_config();url=u+'/storage/v1/object/'+b+'/model-quality/'+digest+'/'
    if raw is not None:
        if len(raw)>MAX_ARCHIVE_BYTES:raise HTTPException(413,'Snapshot exceeds archive limit; select fewer revisions')
        if sha(raw)!=digest:raise ValueError('Archive checksum mismatch')
        hashes=[]
        for i,start in enumerate(range(0,len(raw),ARCHIVE_CHUNK_BYTES)):
            part=raw[start:start+ARCHIVE_CHUNK_BYTES];part_sha=sha(part);hashes.append(part_sha)
            _asset_json(url+f'part-{i:03d}.json',h,{'sha256':part_sha,'base64':base64.b64encode(part).decode()})
        # Publish the index only after every immutable part has been read back.
        _asset_json(url+'manifest.json',h,{'version':1,'sha256':digest,'size':len(raw),'chunks':hashes})
        return raw
    manifest=_asset_json(url+'manifest.json',h)
    if (manifest.get('version')!=1 or manifest.get('sha256')!=digest or type(manifest.get('size')) is not int
        or not 0<=manifest['size']<=MAX_ARCHIVE_BYTES or not isinstance(manifest.get('chunks'),list)
        or len(manifest['chunks'])>MAX_ARCHIVE_BYTES//ARCHIVE_CHUNK_BYTES):
        raise HTTPException(503,'Invalid snapshot archive index')
    parts=[]
    for i,expected in enumerate(manifest['chunks']):
        part=_asset_json(url+f'part-{i:03d}.json',h)
        try:
            encoded=part['base64']
            if not isinstance(encoded,str) or len(encoded)>((ARCHIVE_CHUNK_BYTES+2)//3)*4:raise ValueError()
            decoded=base64.b64decode(encoded,validate=True)
            if sha(decoded)!=expected or part.get('sha256')!=expected:raise ValueError()
        except (ValueError,KeyError,TypeError):raise HTTPException(503,'Snapshot part checksum mismatch') from None
        parts.append(decoded)
    result=b''.join(parts)
    if len(result)!=manifest['size'] or sha(result)!=digest:raise HTTPException(503,'Snapshot archive checksum mismatch')
    return result


def inventory(kind,load=False,selection=None):
    if kind not in ('segmentation','classification'):raise HTTPException(422,'Unknown model type')
    store=QualityStore()
    revisions=store.rows('contour_revisions',order='created_at.desc,correction_id',limit='500')
    if selection is not None:
        wanted={(v['owner_id'],v['correction_id']) for v in selection}
        revisions=[r for r in revisions if (r['owner_id'],r['correction_id']) in wanted]
        if len(revisions)!=len(wanted):raise HTTPException(422,'Selected revisions unavailable in selection window')
    labels=store.rows('ml_taxon_labels',order='created_at.desc,id',limit='1000')
    latest={}
    for l in labels:latest.setdefault((l['owner_id'],l['image_sha256']),l)
    included=[];excluded=[];retained_bytes=0
    for r in revisions:
        label=latest.get((r['owner_id'],r['image_sha256']))
        try:
            if retained_bytes>128*1024*1024:raise ValueError('snapshot_memory_budget_deferred_use_smaller_selection')
            if kind=='segmentation' and r['status']!='accepted':raise ValueError('accepted_revision_required:'+r['status'])
            if kind=='classification' and not label:raise ValueError('separate_confirmed_species_label_required')
            record=_get(r['owner_id'],r['correction_id'])
            s=sample(record,r,{**label['label'],'label_id':label['id']} if label else None,kind)
            included.append(s)
            retained_bytes+=sum(len(v) for v in s.values() if isinstance(v,bytes))
        except (ValueError,KeyError) as e:
            excluded.append({'owner_id':r['owner_id'],'correction_id':r['correction_id'],'analysis_id':r['analysis_id'],
               'status':r['status'],'reason':str(e)[:2000], 'label_id':label['id'] if label else None})
    if load:return included,excluded
    def summary(s):return {k:v for k,v in s.items() if k not in ('image','original','mask','yolo_label')}
    return {'eligible':[summary(s) for s in included],'excluded':excluded,
            'selection_limit':500,'limit_reached':len(revisions)==500,
            'legacy_verified_policy':'excluded_without_provable_revision_decision',
            'classes':{'0':'tree'} if kind=='segmentation' else 'separate_confirmed_taxa',
            'external_classifier':'plantnet_not_finetunable_via_this_API'}


@router.get('/status')
def status(admin=Depends(current_admin)):
    s=QualityStore()
    version=s.request('POST','rpc/model_quality_version',json={})
    from .api import vision
    heartbeat=Path(os.getenv('MODEL_QUALITY_DIR','/app/model-quality'))/'heartbeat'
    age=time.time()-heartbeat.stat().st_mtime if heartbeat.exists() else None
    return {'version':version,'worker':{'online':age is not None and age<30,'heartbeat_age_seconds':age},
      'baseline':{'version':vision.model_version,'classes':vision.model_names,'weights_sha256':getattr(vision,'model_sha256',None)},
      'jobs':s.rows('ml_jobs',order='created_at.desc',limit='30'),
      'snapshots':s.rows('ml_snapshots',order='created_at.desc',limit='30'),
      'models':s.rows('ml_models',order='created_at.desc',limit='30'),
      'active':s.rows('ml_active_models'),
      'policy':{'production_auto_activation':False,'external_species_service':'plantnet',
                'local_classifier':'experimental_separate_scope','training_device':'cpu'}}


@router.get('/data')
def data(model_type:str='segmentation',admin=Depends(current_admin)):return inventory(model_type)


class TaxonInput(BaseModel):
    operation_id:str
    owner_id:str
    correction_id:str=Field(max_length=240)
    parent_id:str|None=None
    scientific_name:str=Field(min_length=2,max_length=200)
    taxon_id:str=Field(min_length=1,max_length=100)
    authority:str=Field(pattern='^(GBIF|POWO|manual_reference)$')
    rank:str=Field(pattern='^(species|genus)$')
    russian_name:str|None=Field(default=None,max_length=200)
    evidence:str=Field(min_length=3,max_length=2000)
    confirmed:bool
    group_id:str|None=Field(default=None,max_length=120)


@router.get('/labels/{owner}/{correction}')
def get_labels(owner:str,correction:str,admin=Depends(current_admin)):
    record=_get(_uuid(owner),correction)
    rows=QualityStore().rows('ml_taxon_labels',owner_id='eq.'+owner,image_sha256='eq.'+record['image_sha256'],order='created_at.desc')
    return {'items':rows}


@router.post('/labels')
def label(body:TaxonInput,admin=Depends(current_admin)):
    owner=_uuid(body.owner_id);record=_get(owner,body.correction_id)
    store=QualityStore()
    # Only the report tied to this already reviewable contour is accessed;
    # this does not grant broad access to unrelated private report history.
    reports=store.rows('report_versions',owner_id='eq.'+owner,analysis_id='eq.'+record['analysis_id'],
        image_sha256='eq.'+record['image_sha256'],order='created_at.desc',limit='1')
    prediction=None
    if reports:prediction=ReportStore().read_blob(owner,reports[0]['payload_sha256']).get('report',{}).get('species')
    fields=body.model_dump(exclude={'operation_id','owner_id','correction_id','parent_id'})
    return store.transition('label',_uuid(body.operation_id),admin,owner_id=owner,
       analysis_id=record['analysis_id'],image_sha256=record['image_sha256'],correction_id=body.correction_id,
       parent_id=_uuid(body.parent_id) if body.parent_id else None,label=fields,original_prediction=prediction)


class SnapshotInput(BaseModel):
    operation_id:str
    model_type:str=Field(pattern='^(segmentation|classification)$')
    seed:int=Field(default=20260917,ge=0,le=2147483647)
    selection:list[dict[str,str]]|None=Field(default=None,max_length=500)


@router.post('/snapshots')
def snapshot(body:SnapshotInput,admin=Depends(current_admin)):
    sid=_uuid(body.operation_id);store=QualityStore()
    selection=body.selection
    if selection is not None:
        if any(set(v)!={'owner_id','correction_id'} or len(v['correction_id'])>240 for v in selection):raise HTTPException(422,'Invalid selection')
        selection=sorted([{'owner_id':_uuid(v['owner_id']),'correction_id':v['correction_id']} for v in selection],key=lambda v:(v['owner_id'],v['correction_id']))
    previous=store.rows('ml_snapshots',id='eq.'+sid,limit='1')
    if previous:
        if previous[0]['actor_id']!=admin or previous[0]['model_type']!=body.model_type or previous[0]['manifest']['seed']!=body.seed or previous[0]['manifest'].get('selection')!=selection:raise HTTPException(409,'Snapshot operation conflict')
        return previous[0]
    samples,excluded=inventory(body.model_type,True,selection)
    samples,duplicates,near=split_samples(samples,body.seed);excluded+=duplicates
    manifest={'version':1,'exporter':EXPORTER_VERSION,'seed':body.seed,'model_type':body.model_type,
       'class_names':['tree'] if body.model_type=='segmentation' else sorted({s['label']['class_key'] for s in samples}),
       'excluded':excluded,'near_duplicates':near,'items':[], 'selection_limit':500,'selection':selection,
       'policy':'accepted_masks_and_separately_confirmed_taxa_no_test_tuning',
       'training_ready':all(any(s['split']==p for s in samples) for p in ('train','val','test'))}
    if body.model_type=='classification':
        manifest['taxonomy']={s['label']['class_key']:{k:s['label'].get(k) for k in ('authority','taxon_id','scientific_name','russian_name','rank')} for s in samples}
        manifest['training_ready']=len(manifest['class_names'])>=2 and all(
            any(s['split']==p and s['label']['class_key']==c for s in samples)
            for c in manifest['class_names'] for p in ('train','val','test'))
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for i,s in enumerate(samples):
            metadata={k:v for k,v in s.items() if k not in ('image','mask','original','yolo_label')}
            metadata['files']={}
            files={f'images/{i}.png':s['image'],f'originals/{i}.bin':s['original']}
            if body.model_type=='segmentation':files.update({f'masks/{i}.png':s['mask'],f'labels/{i}.txt':s['yolo_label'].encode()})
            for name,raw in files.items():
                ti=tarfile.TarInfo(name);ti.size=len(raw);tar.addfile(ti,io.BytesIO(raw))
                metadata['files'][name]=sha(raw)
            manifest['items'].append(metadata)
        raw=canonical(manifest);ti=tarfile.TarInfo('manifest.json');ti.size=len(raw);tar.addfile(ti,io.BytesIO(raw))
    archive=buf.getvalue();digest=sha(archive);asset(digest,archive)
    return store.transition('snapshot',sid,admin,model_type=body.model_type,manifest={**manifest,'archive_sha256':digest})


class JobInput(BaseModel):
    operation_id:str
    snapshot_id:str
    epochs:int=Field(default=1,ge=1,le=5)
    imgsz:int=Field(default=320,ge=320,le=640)


@router.post('/jobs')
def enqueue(body:JobInput,admin=Depends(current_admin)):
    s=QualityStore();job_id=_uuid(body.operation_id)
    existing=s.rows('ml_jobs',id='eq.'+job_id,limit='1')
    if existing:
        old=existing[0]
        if old['actor_id']!=admin or old['snapshot_id']!=_uuid(body.snapshot_id) or any(old['params'].get(k)!=v for k,v in {'epochs':body.epochs,'imgsz':body.imgsz}.items()):
            raise HTTPException(409,'Job operation conflict')
        return old
    snapshot=s.one('ml_snapshots',body.snapshot_id)
    if not snapshot['manifest']['training_ready']:raise HTTPException(422,'Need at least three independent groups: train, validation, test')
    if snapshot['model_type']=='classification':
        items=snapshot['manifest']['items'];classes=snapshot['manifest']['class_names']
        if len(classes)<2 or any(not any(v['split']==p and v['label']['class_key']==c for v in items) for c in classes for p in ('train','val','test')):
            raise HTTPException(422,'Each class needs independent train, validation and test examples')
    base_id=None;base_sha=None
    if snapshot['model_type']=='segmentation':
        from .api import vision
        from .quality_runtime import select_runtime
        runtime=select_runtime(vision)
        base_id=getattr(runtime,'quality_model_id',None)
        base_sha=runtime.model_sha256
    return s.transition('enqueue',job_id,admin,snapshot_id=_uuid(body.snapshot_id),
       params={'epochs':body.epochs,'imgsz':body.imgsz,'batch':1,'workers':0,'device':'cpu','max_seconds':1800,
               'base_model_id':base_id,'base_weights_sha256':base_sha})


@router.post('/jobs/{id}/cancel')
def cancel(id:str,admin=Depends(current_admin)):return QualityStore().transition('cancel',_uuid(id),admin)


class ActivationInput(BaseModel):
    model_id:str|None=None
    expected_generation:int=Field(ge=0)


@router.post('/activate')
def activate(body:ActivationInput,admin=Depends(current_admin)):
    if body.model_id:
        from .quality_runtime import checked_runtime
        checked_runtime(QualityStore().one('ml_models',body.model_id))
    return QualityStore().transition('activate',_uuid(body.model_id) if body.model_id else None,admin,
       expected_generation=body.expected_generation)


@router.get('/models/{id}/diagnostics/{source}/{split}/{index}')
def diagnostics(id:str,source:str,split:str,index:int,admin=Depends(current_admin)):
    from .quality_runtime import model_path
    if source not in ('baseline','candidate') or split not in ('val','test') or index<0:
        raise HTTPException(404,'Diagnostic unavailable')
    model=QualityStore().one('ml_models',id)
    cases=model['metadata'].get('evaluation',{}).get(source+'_'+split,{}).get('cases',[])
    if index>=len(cases):raise HTTPException(404,'Diagnostic unavailable')
    path=model_path(id).parent/'diagnostics'/source/split/f'{index}.png'
    if not path.is_file():raise HTTPException(404,'Diagnostic unavailable')
    if path.stat().st_size>12*1024*1024:raise HTTPException(413,'Diagnostic too large')
    return {'image_base64':base64.b64encode(path.read_bytes()).decode(),'case':cases[index]}
