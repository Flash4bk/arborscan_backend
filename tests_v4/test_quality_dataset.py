import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from arborscan_v4.quality_dataset import polygon_labels, sample, sha, split_samples


def encoded(array):
    out=io.BytesIO(); Image.fromarray(array).save(out,'PNG'); return out.getvalue()


def fixture():
    photo=encoded(np.zeros((64,48,3),np.uint8))
    mask=np.zeros((64,48),np.uint8);mask[2:62,5:44]=255
    png=encoded(mask)
    record={'original_image_base64':base64.b64encode(photo).decode(),
            'mask_png_base64':base64.b64encode(png).decode(),
            'image_sha256':sha(photo),'mask_sha256':sha(png)}
    metadata={'status':'accepted','decisions':[{'action':'accepted','actor_id':'admin','at':'2026-09-18'}],
              'owner_id':'owner','analysis_id':'analysis','correction_id':'revision','image_sha256':sha(photo)}
    return record,metadata


def test_independent_confirmation_and_original_binding():
    record,metadata=fixture()
    assert sample(record,metadata)['fidelity']['iou']==1
    with pytest.raises(ValueError,match='separate_confirmed'):
        sample(record,metadata,model_type='classification')
    metadata['status']='rejected'
    label={'confirmed':True,'rank':'species','scientific_name':'Pinus sylvestris','authority':'GBIF','taxon_id':'123'}
    assert sample(record,metadata,label,'classification')['label']['class_key']=='GBIF:123'
    with pytest.raises(ValueError,match='accepted_revision'):
        sample(record,metadata,label)
    metadata['image_sha256']='x'
    with pytest.raises(ValueError,match='revision_original'):
        sample(record,metadata,label,'classification')


def test_decision_proof_and_smoke_exclusion():
    record,metadata=fixture();metadata['decisions']=[]
    with pytest.raises(ValueError,match='accepted_revision'):
        sample(record,metadata)
    record['purpose']='export_smoke_test_only_not_model_evaluation'
    with pytest.raises(ValueError,match='synthetic_export'):
        sample(record,metadata)


def test_components_and_holes_are_measured_not_silently_removed():
    mask=np.zeros((100,100),np.uint8)
    mask[5:45,5:45]=255;mask[55:95,55:95]=255
    _,f=polygon_labels(mask)
    assert f['components']==2 and f['lost_pixels']==0
    assert f['added_pixels']>0 # Explicit zero-width bridge raster footprint.
    mask[10:40,10:40]=0
    _,f=polygon_labels(mask)
    assert f['holes']==1 and f['iou']<.995
    record,metadata=fixture();record['mask_png_base64']=base64.b64encode(encoded(mask)).decode()
    record['mask_sha256']=sha(encoded(mask))
    with pytest.raises(ValueError,match='dimensions'):
        sample(record,metadata)


def test_thin_branches_edges_and_small_components():
    mask=np.zeros((61,47),np.uint8)
    mask[:,23:25]=255;mask[10,0:47]=255;mask[30,12:36]=255
    _,f=polygon_labels(mask)
    assert f['lost_pixels']==0 and f['iou']==1
    tiny=np.zeros((10,10),np.uint8);tiny[2,2]=255
    with pytest.raises(ValueError,match='degenerate'):
        polygon_labels(tiny)


def test_known_series_and_revisions_never_cross_split():
    rows=[]
    for i in range(12):
        rows.append({'correction_id':str(i),'image_sha256':str(i),'pixels_sha256':str(i),
                     'group_id':'one-tree' if i<3 else None,'near_duplicate_signature':format(i,'064b')})
    rows.append({**rows[0],'correction_id':'new-revision'})
    selected,excluded,near=split_samples(rows,42)
    assert len(selected)==12 and excluded[0]['reason']=='exact_duplicate_or_other_revision'
    assert len({r['split'] for r in selected if r['group_id']=='one-tree'})==1
    assert set(r['split'] for r in selected)=={'train','val','test'}
    assert near and len(selected)==12 # Similarity never silently deletes a photo.
    assert split_samples(rows,42)==(selected,excluded,near)


def test_exif_oriented_dimensions():
    record,metadata=fixture()
    out=io.BytesIO();im=Image.new('RGB',(64,48));exif=im.getexif();exif[274]=6;im.save(out,'JPEG',exif=exif)
    record['original_image_base64']=base64.b64encode(out.getvalue()).decode()
    record['image_sha256']=metadata['image_sha256']=sha(out.getvalue())
    result=sample(record,metadata)
    assert (result['width'],result['height'])==(48,64)
