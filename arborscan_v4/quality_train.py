"""Isolated training subprocess, not an API request. No shell arguments accepted."""
import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
import cv2
import numpy as np
from .quality_dataset import sha
from .model_quality_api import QualityStore,asset
from .quality_runtime import model_path


def verify_training_masks(manifest,data,imgsz):
    """Use the installed YOLO decoder/resampler/rasterizer, not our exporter alone."""
    from ultralytics.utils.ops import resample_segments
    from ultralytics.data.utils import polygon2mask
    rows=[]
    for item in manifest['items']:
        label=next(k for k in item['files'] if k.startswith('labels/'))
        mask=next(k for k in item['files'] if k.startswith('masks/'))
        truth=cv2.imread(str(data/mask),0)>127;h,w=truth.shape
        points=np.array((data/label).read_text().split()[1:],np.float32).reshape(-1,2)
        points=resample_segments([points],max(1000,len(points)+1))[0]
        original=points*np.array([w,h],np.float32)
        raster=polygon2mask((h,w),[original.reshape(-1)],downsample_ratio=1)>0
        def metrics(a,b):
            return {'iou':float((a&b).sum()/max(1,(a|b).sum())),
                    'lost_pixels':int((a&~b).sum()),'added_pixels':int((~a&b).sum())}
        fidelity=metrics(truth,raster)
        # Letterbox scale preserves aspect; no training geometric augmentation.
        scale=imgsz/max(h,w);nw,nh=round(w*scale),round(h*scale)
        scaled_truth=cv2.resize(truth.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)>0
        scaled_raster=polygon2mask((nh,nw),[(points*np.array([nw,nh],np.float32)).reshape(-1)],downsample_ratio=1)>0
        row={'correction_id':item['correction_id'],'original':fidelity,'training_resolution':metrics(scaled_truth,scaled_raster)}
        rows.append(row)
    return rows


def unpack(snapshot,root):
    raw=asset(snapshot['manifest']['archive_sha256'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for member in tar.getmembers():
            parts=Path(member.name).parts
            if member.name.startswith('/') or '..' in parts or not member.isfile():raise ValueError('Unsafe snapshot archive')
            path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['items']:
        for name,digest in item['files'].items():
            if sha((root/name).read_bytes())!=digest:raise ValueError('Dataset checksum mismatch')
    if manifest!={k:v for k,v in snapshot['manifest'].items() if k!='archive_sha256'}:raise ValueError('Manifest mismatch')
    return manifest


def segmentation_metrics(runtime,items,root,out):
    rows=[]
    for i,item in enumerate(items):
        image_name=next(k for k in item['files'] if k.startswith('images/'))
        mask_name=next(k for k in item['files'] if k.startswith('masks/'))
        image=cv2.imread(str(root/image_name));truth=cv2.imread(str(root/mask_name),0)>127
        started=time.perf_counter();prediction=runtime.infer(image)
        seconds=time.perf_counter()-started
        predicted=prediction.mask>127 if prediction.detected else np.zeros_like(truth)
        tp=int((truth & predicted).sum());fp=int((~truth & predicted).sum());fn=int((truth & ~predicted).sum())
        row={'correction_id':item['correction_id'],'split':item['split'],'class':'tree','iou':tp/max(1,tp+fp+fn),
          'dice':2*tp/max(1,2*tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'seconds':seconds,
          'complexity':item.get('fidelity')}
        overlay=image.copy();overlay[truth & ~predicted]=(0,0,255);overlay[predicted & ~truth]=(255,0,0)
        overlay[truth & predicted]=(0,180,0)
        visual=cv2.addWeighted(image,.5,overlay,.5,0)
        scale=min(1,1280/max(visual.shape[:2]))
        if scale<1:visual=cv2.resize(visual,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out/f'{i}.png'),visual)
        rows.append(row)
    return {'cases':rows,'classes':{'tree':{'mean_iou':float(np.mean([r['iou'] for r in rows])) if rows else None}},
            'seconds_total':sum(r['seconds'] for r in rows)}


def classification_metrics(model,items,root,classes):
    matrix=np.zeros((len(classes),len(classes)),dtype=int);times=[]
    for item in items:
        name=next(k for k in item['files'] if k.startswith('images/'))
        start=time.perf_counter();pred=model(str(root/name),verbose=False,device='cpu')[0].probs.top1;times.append(time.perf_counter()-start)
        matrix[classes.index(item['label']['class_key']),int(pred)]+=1
    per=[]
    for i,c in enumerate(classes):
        tp=int(matrix[i,i]);fp=int(matrix[:,i].sum()-tp);fn=int(matrix[i,:].sum()-tp)
        per.append({'class':c,'support':int(matrix[i,:].sum()),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)})
    return {'macro_f1':float(np.mean([r['f1'] for r in per])),'per_class':per,'confusion_matrix':matrix.tolist(),
       'unknown_species':'not_evaluated_without_independent_unknown_examples','seconds_total':sum(times),
       'baseline_comparison':'not_comparable_to_external_PlantNet_taxonomy'}


def train(job_id):
    import resource
    import torch
    from ultralytics import YOLO
    from .vision_engine import TreeVisionRuntime
    torch.set_num_threads(1)
    store=QualityStore();job=store.one('ml_jobs',job_id);snapshot=store.one('ml_snapshots',job['snapshot_id'])
    root=model_path(job_id).parent;root.mkdir(parents=True,exist_ok=True)
    def stage(name):
        temporary=root/'stage.tmp';temporary.write_text(json.dumps({'stage':name}));temporary.replace(root/'stage.json')
    stage('snapshot_verification')
    data=root/'data';data.mkdir(exist_ok=True);manifest=unpack(snapshot,data)
    kind=snapshot['model_type'];classes=manifest['class_names'];params=job['params']
    base=model_path(params['base_model_id']) if params.get('base_model_id') else Path(os.getenv('MODEL_QUALITY_BASE_WEIGHTS','/app/models/model_v4.pt'))
    initial_sha=sha(base.read_bytes()) if kind=='segmentation' else None
    if kind=='segmentation' and initial_sha!=params.get('base_weights_sha256'):
        raise ValueError('Pinned baseline checksum mismatch')
    dataset=root/'dataset'
    for i,item in enumerate(manifest['items']):
        name=next(k for k in item['files'] if k.startswith('images/'));split=item['split']
        if kind=='segmentation':
            dst=dataset/'images'/split/f'{i}.png';label=dataset/'labels'/split/f'{i}.txt';label.parent.mkdir(parents=True,exist_ok=True)
            label.write_bytes((data/next(k for k in item['files'] if k.startswith('labels/'))).read_bytes())
        else:dst=dataset/split/f'{classes.index(item["label"]["class_key"]):05d}'/f'{i}.png'
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(data/name,dst)
    if kind=='segmentation':
        stage('mask_export_verification')
        audit=verify_training_masks(manifest,data,params['imgsz'])
        (root/'mask-fidelity.json').write_text(json.dumps(audit))
        if any(r['original']['iou']<.995 or r['training_resolution']['iou']<.95 for r in audit):
            raise ValueError('training_mask_representation_loss')
        model=YOLO(str(base))
        yaml=dataset/'data.yaml';yaml.write_text(f'path: {dataset}\ntrain: images/train\nval: images/val\n'+'names: [tree]\n')
        train_data=str(yaml)
    else:
        model=YOLO('yolo11n-cls.yaml');train_data=str(dataset)
    started=time.perf_counter()
    stage('training')
    model.train(data=train_data,epochs=params['epochs'],imgsz=params['imgsz'],batch=1,workers=0,device='cpu',
       project=str(root/'runs'),name='candidate',exist_ok=True,seed=manifest['seed'],deterministic=True,plots=False,
       save=True,pretrained=(kind=='segmentation'),amp=False,
       **({'mask_ratio':1,'mosaic':0,'mixup':0,'copy_paste':0,'degrees':0,'translate':0,'scale':0,
           'fliplr':0,'flipud':0,'perspective':0} if kind=='segmentation' else {}))
    weights=root/'runs/candidate/weights/best.pt'
    if not weights.is_file():raise ValueError('Trainer produced no best weights')
    shutil.copyfile(weights,root/'candidate.pt')
    metadata={'model_type':kind,'architecture':'ultralytics_yolo_segment' if kind=='segmentation' else 'yolo11n_classify_from_scratch',
       'class_names':classes,'weights_sha256':sha(weights.read_bytes()),'base_weights_sha256':initial_sha if kind=='segmentation' else None,
       'snapshot_id':snapshot['id'],'archive_sha256':manifest.get('archive_sha256',snapshot['manifest']['archive_sha256']),
       'params':params,'training_seconds':time.perf_counter()-started,'preprocessing':'exif_rgb_to_bgr_v1',
       'postprocessing':'tree_selected_instance_v1' if kind=='segmentation' else 'closed_set_top1',
       'compatible':False,'eligible_for_activation':False,'status':'experimental',
       'test_policy':'held_out_not_used_for_training_or_best_epoch_selection'}
    if kind=='segmentation':metadata['training_mask_fidelity']=audit
    else:metadata['taxonomy']=manifest['taxonomy']
    validation=[i for i in manifest['items'] if i['split']=='val'];test=[i for i in manifest['items'] if i['split']=='test']
    stage('held_out_evaluation')
    if kind=='segmentation':
        candidate=TreeVisionRuntime();candidate._select_model_path=lambda:(root/'candidate.pt',4);candidate.load()
        if candidate.model_names!={0:'tree'}:raise ValueError('Candidate classes incompatible')
        baseline=TreeVisionRuntime();baseline._select_model_path=lambda:(base,4);baseline.load()
        metadata['evaluation']={}
        for name,runtime in [('baseline',baseline),('candidate',candidate)]:
            for split,items in [('val',validation),('test',test)]:
                out=root/'diagnostics'/name/split;out.mkdir(parents=True,exist_ok=True)
                metadata['evaluation'][name+'_'+split]=segmentation_metrics(runtime,items,data,out)
        metadata['compatible']=True
        enough=all(len({i['split_group'] for i in manifest['items'] if i['split']==p})>=n for p,n in [('train',30),('val',10),('test',10)])
        better=metadata['evaluation']['candidate_val']['classes']['tree']['mean_iou']>=metadata['evaluation']['baseline_val']['classes']['tree']['mean_iou']
        metadata['eligible_for_activation']=enough and better and not any(p['cross_split'] for p in manifest['near_duplicates'])
        metadata['gate']='>=30 train/10 val/10 test independent images; no unresolved cross-split near duplicates; validation mean IoU non-regression. Minimum operational gate, not statistical proof.'
    else:
        candidate=YOLO(str(root/'candidate.pt'))
        metadata['evaluation']={'validation':classification_metrics(candidate,validation,data,classes),'test':classification_metrics(candidate,test,data,classes)}
        metadata['compatible']=True
        metadata['gate']='Local closed-set classifier remains experimental and never replaces PlantNet'
    metadata['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (root/'result.json').write_text(json.dumps(metadata),encoding='utf8')


if __name__=='__main__':
    import sys
    train(sys.argv[1])
