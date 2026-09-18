import hashlib
import sys
import types
from unittest.mock import patch
import pytest
from fastapi import HTTPException
from arborscan_v4 import quality_runtime as r


def test_incompatible_corrupt_weights_and_pinned_request(tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_QUALITY_DIR',str(tmp_path));r._cache.clear()
    loads=[]
    class Runtime:
        def load(self):
            self.model_names={0:'tree'};loads.append(self._select_model_path()[0])
        def infer(self,image):return None
    monkeypatch.setitem(sys.modules,'arborscan_v4.vision_engine',types.SimpleNamespace(TreeVisionRuntime=Runtime))
    key='00000000-0000-4000-8000-000000000001'
    path=r.model_path(key);path.parent.mkdir(parents=True);path.write_bytes(b'synthetic-not-weights')
    row={'id':key,'model_type':'segmentation','metadata':{'class_names':['tree'],'preprocessing':'exif_rgb_to_bgr_v1','weights_sha256':'bad'}}
    with pytest.raises(HTTPException):r.checked_runtime(row)
    assert not loads
    row['metadata']['weights_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    row['metadata']['class_names']=['trunk','crown']
    with pytest.raises(HTTPException):r.checked_runtime(row)
    assert not loads
    row['metadata']['class_names']=['tree'];first=r.checked_runtime(row)
    assert r.checked_runtime(row) is first and len(loads)==1
    # Switching the shared pointer only affects the next request; captured runtime stays.
    monkeypatch.setenv('MODEL_QUALITY_ENABLED','1');default=object()
    with patch('arborscan_v4.model_quality_api.QualityStore') as store:
        store.return_value.rows.return_value=[{'model_id':key}];store.return_value.one.return_value=row
        captured=r.select_runtime(default)
        store.return_value.rows.return_value=[{'model_id':None}]
        assert r.select_runtime(default) is default and captured is first
    path.write_bytes(b'corrupted-after-cache')
    with pytest.raises(HTTPException):r.checked_runtime(row)
    r._cache.clear()
