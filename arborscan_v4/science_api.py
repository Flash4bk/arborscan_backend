"""Owner-private, immutable scientific scenarios using the report-version store.

No training queue, external inference, species default or production migration.
The supplied parameters are user scenarios, not server-verified field inputs.
"""
import copy
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from arborscan_science import section
from arborscan_science.section import ScienceInputError, section_capacity
from .corrections_api import current_user, _uuid
from .report_history import ReportStore, canonical


router = APIRouter(prefix='/v4/science', tags=['private scientific scenarios'])
MAX_INPUT_BYTES = 12_000


def save_section_result(owner, source_version, result_version, inputs, store=None):
    """A retry uses the same immutable version ID and deterministic payload.

    The existing transactional index enforces one child per parent and compares
    content on repeated version IDs. An unindexed blob is never a saved result.
    """
    owner, source_version, result_version = map(_uuid, (owner, source_version, result_version))
    if source_version == result_version:
        raise HTTPException(422, 'A calculation requires a new report version')
    store = store if store is not None else ReportStore()
    original = store.get(owner, source_version)
    row = original['record']
    if row.get('owner_id') != owner or row.get('version_id') != source_version:
        raise HTTPException(404, 'Report not found')
    try:
        input_bytes = canonical(inputs)
        if len(input_bytes) > 8192:
            raise ValueError()
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise HTTPException(422, 'Invalid scientific inputs') from None
    # Reuse committed content before recomputation: an upgrade may change the
    # implementation fingerprint after the first response was lost in transit.
    try:
        existing = store.get(owner, result_version)
    except HTTPException as error:
        if error.status_code != 404:
            raise
    else:
        saved_row = existing['record']
        saved_result = existing['snapshot'].get('science_result')
        if (not isinstance(saved_result, dict) or
                saved_row.get('owner_id') != owner or saved_row.get('parent_id') != source_version or
                saved_row.get('analysis_id') != row['analysis_id'] or
                saved_row.get('image_sha256') != row['image_sha256'] or
                saved_result.get('source_payload_sha256') != row['payload_sha256'] or
                saved_result.get('source_version_id') != source_version or
                saved_result.get('result_id') != result_version or
                saved_result.get('method_id') != section.METHOD_ID or
                saved_result.get('input_sha256') != hashlib.sha256(input_bytes).hexdigest()):
            raise HTTPException(409, 'Calculation intent changed; retain the draft and reopen the latest version')
        return {'saved': True, 'persisted': True, 'record': saved_row,
                'science_version': 1, 'science_result': saved_result}
    try:
        result = section_capacity(inputs)
    except ScienceInputError as error:
        raise HTTPException(422, error.detail()) from None
    payload = copy.deepcopy(original['snapshot'])
    if payload.get('image', {}).get('sha256') != row['image_sha256']:
        raise HTTPException(503, 'Original photo binding is unavailable')
    result.update(
        result_id=result_version,
        source_version_id=source_version,
        source_payload_sha256=row['payload_sha256'],
        image_sha256=row['image_sha256'],
        correction_id=payload.get('correction_id'),
        implementation_sha256=hashlib.sha256(Path(section.__file__).read_bytes()).hexdigest(),
        input_verification='user_supplied_not_field_verified',
    )
    result['result_sha256'] = hashlib.sha256(canonical(result)).hexdigest()
    # This top-level field cannot be assigned by the ordinary snapshot upload.
    # Physical measurements and previous versions remain unchanged.
    payload.update(science_result=result, change_source='science_scenario')
    saved = store.save(owner, row['analysis_id'], result_version, source_version, payload)
    if (not isinstance(saved, dict) or saved.get('saved') is not True or
            saved.get('persisted') is not True or
            saved.get('record', {}).get('version_id') != result_version):
        raise HTTPException(503, 'Calculation save was not confirmed; retry the same operation')
    return {**saved, 'science_version': 1, 'science_result': result}


@router.get('/capabilities')
def capabilities(owner=Depends(current_user)):
    ReportStore().ready()
    return {'science_version': 1, 'methods': [section.METHOD_ID],
            'validation_status': 'scenario_software_only',
            'photo_beta_available': False, 'field_validated': False}


@router.post('/section-capacity')
async def calculate_section(request: Request, owner=Depends(current_user)):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_INPUT_BYTES:
            raise HTTPException(413, 'Scientific inputs exceed 12 KiB')
    try:
        data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(data, dict) or set(data) != {'source_version_id', 'version_id', 'inputs'}:
            raise ValueError()
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise HTTPException(422, 'Invalid scientific request') from None
    return await run_in_threadpool(save_section_result, owner, data['source_version_id'],
                                  data['version_id'], data['inputs'])
