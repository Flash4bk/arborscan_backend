"""Offline validation for a future contour-to-dataset adapter, NOT that adapter.

The caller must obtain records from trusted server snapshots, never client JSON.
This gate selects no samples, downloads nothing and cannot trigger training.
"""
import base64
import hashlib
import cv2
import numpy as np


def validate_selection(records):
    """Reject unreviewed revisions and duplicate source images, even re-encoded.

    Each record includes split=train|val, correction_id, review_status,
    decisions, image_sha256, mask_sha256, original_image_base64, mask_png_base64.
    Exactly one accepted revision per image is allowed across the entire export.
    """
    groups, ids = set(), set()
    for record in records:
        if record.get('split') not in ('train','val') or record.get('review_status') != 'accepted':
            raise ValueError('Only accepted segmentation revisions may be selected')
        history = record.get('decisions', [])
        if not history or history[-1].get('action') != 'accepted' or not history[-1].get('actor_id') or not history[-1].get('at'):
            raise ValueError('Acceptance audit is required')
        key = (record.get('owner_id'), record.get('correction_id'))
        if not all(key) or key in ids: raise ValueError('Duplicate or incomplete revision identity')
        ids.add(key)
        image_raw = base64.b64decode(record['original_image_base64'],validate=True)
        mask_raw = base64.b64decode(record['mask_png_base64'],validate=True)
        if hashlib.sha256(image_raw).hexdigest()!=record['image_sha256'] or hashlib.sha256(mask_raw).hexdigest()!=record['mask_sha256']:
            raise ValueError('Object digest mismatch')
        if not mask_raw.startswith(b'\x89PNG\r\n\x1a\n'): raise ValueError('Mask must be PNG')
        image = cv2.imdecode(np.frombuffer(image_raw,np.uint8),cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(mask_raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2]!=mask.shape:
            raise ValueError('Image/mask dimensions differ')
        foreground=int(np.count_nonzero(mask>127))
        if foreground==0 or foreground==mask.size: raise ValueError('Degenerate mask')
        # Group by oriented decoded pixels as well as source bytes: PNG/JPEG
        # containers carrying identical pixels cannot cross the split boundary.
        group=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        if group in groups: raise ValueError('Multiple revisions of one image in export')
        groups.add(group)
    return {'revision_count':len(ids),'image_groups':len(groups), 'training_connected':False}
