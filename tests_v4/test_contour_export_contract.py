import base64
import hashlib
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('contract',Path(__file__).resolve().parents[1]/'tools/contour_export_contract.py')
contract=importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)

def record(key='one',split='train'):
    image=cv2.imencode('.png',np.full((32,32,3),150,np.uint8))[1].tobytes()
    m=np.zeros((32,32),np.uint8); m[5:25,10:20]=255
    mask=cv2.imencode('.png',m)[1].tobytes()
    return {'owner_id':'owner','correction_id':key,'split':split,'review_status':'accepted',
        'decisions':[{'action':'accepted','actor_id':'admin','at':'2026-09-16T00:00:00Z'}],
        'original_image_base64':base64.b64encode(image).decode(), 'mask_png_base64':base64.b64encode(mask).decode(),
        'image_sha256':hashlib.sha256(image).hexdigest(),'mask_sha256':hashlib.sha256(mask).hexdigest()}

def test_selected_accepted_revision_does_not_connect_training():
    assert contract.validate_selection([record()])['training_connected'] is False

def test_revisions_cannot_cross_train_validation_or_duplicate_within_split():
    for split in ['train','val']:
        with pytest.raises(ValueError,match='Multiple revisions'):
            contract.validate_selection([record(),record('two',split)])

def test_rejected_mask_or_forged_digest_fails_closed():
    for changes in [{'review_status':'rejected'},{'decisions':[]},{'image_sha256':'wrong'}]:
        with pytest.raises(ValueError): contract.validate_selection([{**record(),**changes}])
