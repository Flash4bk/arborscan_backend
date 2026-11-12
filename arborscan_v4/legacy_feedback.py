"""Bridge owner-authenticated legacy feedback to immutable v4 moderation.

This module has no API/model startup side effects. Scientific corrections remain
unverified proposal fields; it never edits an analysis or a prior decision.
"""
import io
import json
import math

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError


def validate_feedback_proposal(value):
    try:
        if (not isinstance(value, dict) or set(value) != {
                'version', 'kind', 'training_consent', 'quality', 'corrections', 'has_user_mask'}
                or type(value['version']) is not int or value['version'] != 1
                or value['kind'] != 'legacy_user_feedback'
                or type(value['training_consent']) is not bool
                or type(value['has_user_mask']) is not bool):
            raise ValueError()
        quality = value['quality']
        if (not isinstance(quality, dict)
                or set(quality) != {'tree_ok', 'stick_ok', 'params_ok', 'species_ok'}
                or any(type(v) is not bool for v in quality.values())):
            raise ValueError()
        corrections = value['corrections']
        if (not isinstance(corrections, dict) or set(corrections) != {
                'species', 'height_m', 'crown_width_m', 'trunk_diameter_m', 'scale_px_to_m'}):
            raise ValueError()
        name = corrections['species']
        if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 200):
            raise ValueError()
        for key, limits in [('height_m', (.5, 100)), ('crown_width_m', (.1, 100)),
                            ('trunk_diameter_m', (.01, 10)), ('scale_px_to_m', (0, float('inf')))]:
            number = corrections[key]
            if number is not None and (type(number) not in (int, float)
                    or not math.isfinite(number) or number <= 0 or not limits[0] <= number <= limits[1]):
                raise ValueError()
        # Return plain JSON with bounded fields, not an alias to a caller's map.
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if len(raw.encode()) > 12_000:
            raise ValueError()
        return json.loads(raw)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise HTTPException(422, 'Invalid legacy feedback proposal') from None


def _normalized_mask(image, mask):
    from .corrections_api import MAX_IMAGE, MAX_MASK, MAX_PIXELS
    if not image or not mask or len(image) > MAX_IMAGE or len(mask) > MAX_MASK:
        raise HTTPException(413, 'Photo or mask exceeds upload size limit')
    try:
        with Image.open(io.BytesIO(image)) as photo:
            if photo.width * photo.height > MAX_PIXELS:
                raise HTTPException(413, 'Photo exceeds 25 megapixels')
            size = photo.size
            if photo.getexif().get(274, 1) in (5, 6, 7, 8):
                size = size[::-1]
        with Image.open(io.BytesIO(mask)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(413, 'Mask exceeds 25 megapixels')
            raster = ImageOps.exif_transpose(source).convert('L')
            if raster.size != size:
                raster = raster.resize(size, Image.Resampling.NEAREST)
            binary = np.where(np.asarray(raster) > 127, 255, 0).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(binary).save(output, 'PNG')
        raw = output.getvalue()
        if len(raw) > MAX_MASK:
            raise HTTPException(413, 'Mask exceeds upload size limit')
        return raw
    except Image.DecompressionBombError:
        raise HTTPException(413, 'Image exceeds supported resolution') from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, 'Photo or mask cannot be decoded') from None


def save_feedback_proposal(owner, analysis_id, image, mask, proposal):
    # Lazy import keeps server.py -> adapter -> corrections free of ML startup.
    from . import corrections_api as api
    proposal = validate_feedback_proposal(proposal)
    mask = _normalized_mask(image, mask)
    store = api.WorkflowStore(api._config)
    store.ready()
    saved = api._validate_and_save(owner, analysis_id, image, mask, proposal=proposal)
    key = saved['correction_id']
    record = api._get(owner, key)
    if record.get('feedback_proposal') != proposal or not api._requires_workflow_commit(record):
        raise HTTPException(503, 'Feedback proposal was not confirmed; retry the same request')
    metadata = store.transition('legacy', owner, key, analysis=record['analysis_id'],
                                image=record['image_sha256'])
    return {'saved': True, 'correction_id': key, 'review_status': metadata['status'],
            'workflow_version': 1, 'verified': False,
            'eligible_for_training': False}
