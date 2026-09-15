"""Transactional revision metadata; immutable images stay in the private bucket.

No writes to verified datasets, training flags, physical measurements or models.
"""
import json
import math
import requests
from fastapi import HTTPException


def validate_editor(raw, width, height):
    if not isinstance(raw, str) or len(raw.encode()) > 256_000:
        raise HTTPException(422, 'Editor state exceeds limit')
    try:
        state = json.loads(raw)
        if set(state) != {'version', 'coordinates', 'width', 'height', 'closed', 'points'}:
            raise ValueError()
        if (type(state['version']) is not int or state['version'] != 1 or
            state['coordinates'] != 'normalized_oriented_image' or
            type(state['width']) is not int or type(state['height']) is not int or
            state['width'] != width or state['height'] != height or state['closed'] is not True):
            raise ValueError()
        points = state['points']
        if not isinstance(points, list) or not 3 <= len(points) <= 4096:
            raise ValueError()
        for p in points:
            if not isinstance(p, dict) or set(p) != {'x','y'}:
                raise ValueError()
            if any(type(p[k]) not in (int,float) or not math.isfinite(p[k]) or not 0 <= p[k] <= 1 for k in ('x','y')):
                raise ValueError()
        if len({(p['x'],p['y']) for p in points}) < 3:
            raise ValueError()
        return state
    except (TypeError, ValueError, KeyError):
        raise HTTPException(422, 'Invalid editor state or image coordinates') from None


class WorkflowStore:
    def __init__(self, config):
        self.url, self.headers, _ = config()

    def request(self, method, path, **kwargs):
        try:
            res = requests.request(method, self.url + '/rest/v1/' + path,
                                   headers=self.headers, timeout=30, **kwargs)
            if res.status_code >= 400:
                code = res.json().get('code')
                if code in ('23505', 'P0001'):
                    raise HTTPException(409, 'Revision changed; reopen latest revision. Draft retained.')
                if code == '42501': raise HTTPException(403, 'Admin required')
                if code == 'P0002': raise HTTPException(404, 'Revision not found')
                raise HTTPException(503, 'Contour workflow migration is not ready')
            return res.json()
        except (requests.RequestException, ValueError):
            raise HTTPException(503, 'Contour workflow is unavailable') from None

    def ready(self):
        self.request('GET', 'contour_revisions', params={'select':'correction_id','limit':'0'})

    def get(self, owner, key):
        rows = self.request('GET', 'contour_revisions', params={
            'owner_id':'eq.'+owner, 'correction_id':'eq.'+key, 'limit':'1'})
        if not rows: return None
        children = self.request('GET', 'contour_revisions', params={
            'owner_id':'eq.'+owner, 'parent_id':'eq.'+key, 'select':'correction_id', 'limit':'1'})
        return {**rows[0], 'next_revision_id':children[0]['correction_id'] if children else None}

    def transition(self, action, owner, key, **fields):
        return self.request('POST', 'rpc/contour_transition', json={
            'p_action':action, 'p_owner':owner, 'p_id':key, **{'p_'+k:v for k,v in fields.items()}})

    def queue(self, offset):
        rows = self.request('GET','contour_revisions',params={
            'status':'eq.submitted','order':'created_at.asc,correction_id.asc,owner_id.asc',
            'offset':str(offset),'limit':'50'})
        return {'items':rows, 'next_offset':offset+len(rows) if len(rows)==50 else None}


def overlay(record, metadata):
    return {**record, 'workflow_version':1, 'review_status':metadata['status'],
            'parent_id':metadata.get('parent_id'), 'decisions':metadata['decisions'],
            'next_revision_id':metadata.get('next_revision_id'),
            'eligible_for_training':False}
