import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from arborscan_v4 import corrections_api as api

OWNER = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
OTHER = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
KEY = OWNER + '_' + 'a' * 64 + '.json'
PHOTO = (Path(__file__).parents[1] / 'arborscan_app/test/fixtures/reference_exif6.jpg').read_bytes()


@pytest.fixture(autouse=True)
def private_config():
    # Constructor metadata only; every storage read is explicitly mocked.
    with patch.object(api, '_config', return_value=('https://example.invalid', {}, 'private')):
        yield


def record(**extra):
    return {'schema_version': 1, 'owner_id': OWNER,
            'original_image_base64': base64.b64encode(PHOTO).decode(), **extra}


def app():
    application = FastAPI()
    application.include_router(api.router)
    return application


def test_preview_routes_deny_missing_auth_before_storage():
    client = TestClient(app())
    with patch.object(api, '_get') as read:
        for url in [f'/v4/corrections/{KEY}/preview',
                    f'/v4/corrections/workflow/review/{OWNER}/{KEY}/preview']:
            assert client.get(url).status_code == 401
    read.assert_not_called()


def test_owner_preview_is_bounded_oriented_and_contains_no_original_or_mask():
    application = app()
    application.dependency_overrides[api.current_user] = lambda: OWNER
    stored = record()
    with patch.object(api, '_get', return_value=stored) as read, patch.object(api.WorkflowStore, 'get', return_value=None):
        response = TestClient(application).get(f'/v4/corrections/{KEY}/preview')
    read.assert_called_once_with(OWNER, KEY)
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'private, no-store'
    value = response.json()
    assert set(value) == {'correction_id', 'content_type', 'thumbnail_base64', 'width', 'height'}
    assert value['correction_id'] == KEY
    assert (value['width'], value['height']) == (80, 160)
    raw = base64.b64decode(value['thumbnail_base64'])
    assert len(raw) <= 64 * 1024
    with Image.open(BytesIO(raw)) as preview:
        assert preview.format == 'JPEG'
        assert preview.size == (80, 160)
    assert stored['original_image_base64'] == base64.b64encode(PHOTO).decode()


def test_other_user_cannot_read_preview_by_known_identifier():
    application = app()
    application.dependency_overrides[api.current_user] = lambda: OTHER
    def read(owner, key):
        assert owner == OTHER
        raise HTTPException(404, 'not found')
    with patch.object(api, '_get', side_effect=read):
        assert TestClient(application).get(f'/v4/corrections/{KEY}/preview').status_code == 404


def test_admin_preview_uses_same_committed_owner_record_and_regular_user_denied():
    application = app()
    def deny():
        raise HTTPException(403, 'Admin required')
    application.dependency_overrides[api.current_admin] = deny
    client = TestClient(application)
    path = f'/v4/corrections/workflow/review/{OWNER}/{KEY}/preview'
    with patch.object(api, '_get') as read:
        assert client.get(path).status_code == 403
    read.assert_not_called()
    application.dependency_overrides[api.current_admin] = lambda: OTHER
    with patch.object(api, '_get', return_value=record()) as read, patch.object(api.WorkflowStore, 'get', return_value={'status': 'submitted', 'decisions': []}):
        assert client.get(path).status_code == 200
    read.assert_called_once_with(OWNER, KEY)


@pytest.mark.parametrize('extra', [{'schema_version': 2}, {'requires_workflow_commit': True}, {'feedback_proposal': {'version': 1}}])
def test_uncommitted_revision_or_feedback_cannot_be_previewed(extra):
    application = app()
    application.dependency_overrides[api.current_user] = lambda: OWNER
    with patch.object(api, '_get', return_value=record(**extra)), patch.object(api.WorkflowStore, 'get', return_value=None):
        assert TestClient(application).get(f'/v4/corrections/{KEY}/preview').status_code == 409


def test_preview_validates_size_pixels_and_decode_without_writing_storage():
    with patch.object(api, 'MAX_IMAGE', 1):
        with pytest.raises(HTTPException) as error:
            api._thumbnail(record(), KEY)
        assert error.value.status_code == 413
    with patch.object(api, 'MAX_PIXELS', 1), patch('PIL.ImageOps.exif_transpose') as decode:
        with pytest.raises(HTTPException) as error:
            api._thumbnail(record(), KEY)
        assert error.value.status_code == 413
        decode.assert_not_called()
    with pytest.raises(HTTPException) as error:
        api._thumbnail(record(original_image_base64='not-base64'), KEY)
    assert error.value.status_code == 422
