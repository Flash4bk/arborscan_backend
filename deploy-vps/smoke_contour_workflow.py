"""Explicit production smoke test using two newly registered synthetic accounts.

Uses ordinary HTTPS authentication for all app requests. Only the second freshly
registered fixture is granted admin through server credentials. Deletes ONLY
these fixtures on exit. Never prints credentials, response bodies or owner IDs.
Run inside v4: python - < deploy-vps/smoke_contour_workflow.py
"""
import concurrent.futures
import json
import secrets
import sys
import uuid
from urllib.parse import quote
import cv2
import numpy as np
import requests
from arborscan_v4.corrections_api import _config

V3 = 'https://31.57.170.88/api/v3'
V4 = 'https://31.57.170.88/api/v4/v4/corrections'


def main():
    url, service, bucket = _config()
    accounts = []
    stage = 'registration'
    def db(method, table, **kwargs):
        r = requests.request(method, url+'/rest/v1/'+table, headers=service, timeout=30, **kwargs)
        assert r.status_code < 300, 'Fixture database operation failed'
        return r
    def req(method, suffix='', token=None, expected=200, **kwargs):
        r = requests.request(method, V4+suffix, headers={'Authorization':'Bearer '+token} if token else {}, timeout=60, **kwargs)
        assert r.status_code == expected, 'Unexpected API status '+str(r.status_code)
        return r.json()
    try:
        for _ in range(2):
            email = 'contour-smoke-'+uuid.uuid4().hex+'@example.invalid'
            r = requests.post(V3+'/auth/register', json={'name':'Contour workflow smoke fixture',
                'email':email,'password':secrets.token_urlsafe(32)}, timeout=60)
            assert r.status_code == 200, 'Fixture registration failed'
            a = r.json()
            assert a['user']['email'] == email
            accounts.append({'id':a['user']['id'], 'token':a['token'], 'email':email})
        owner, admin = accounts
        stage = 'server role enforcement'
        req('GET','/workflow/queue',owner['token'],expected=403)
        db('PATCH','users',params={'id':'eq.'+admin['id'],'email':'eq.'+admin['email']},json={'role':'admin'})
        assert req('GET','/workflow/capabilities',owner['token'])['workflow_version'] == 1
        image = cv2.imencode('.png',np.full((64,48,3),150,np.uint8))[1].tobytes()
        pixels = np.zeros((64,48),np.uint8); pixels[6:54,10:30]=255
        mask = cv2.imencode('.png',pixels)[1].tobytes()
        aid = str(uuid.uuid4())
        state = {'version':1,'coordinates':'normalized_oriented_image','width':48,'height':64,
            'closed':True,'points':[{'x':.2,'y':.1},{'x':.8,'y':.1},{'x':.5,'y':.8}]}
        def save(parent=None, x=.2, expected=200, legacy=False):
            s = json.loads(json.dumps(state)); s['points'][0]['x'] = x
            data = {'analysis_id':aid}
            if not legacy: data['editor_state'] = json.dumps(s)
            if parent: data['parent_id'] = parent
            return req('POST','' if legacy else '/workflow',owner['token'],expected,
                data=data,files={'image':('photo.png',image),'mask':('mask.png',mask)})
        stage = 'legacy PNG and revision retries'
        old = save(legacy=True)['correction_id']
        assert 'editor_state' not in req('GET','/'+old,owner['token'])
        first = save(); key = first['correction_id']
        assert first['saved'] is True and save()['correction_id'] == key
        assert req('GET','/'+key,owner['token'])['editor_state'] == state
        req('GET','/'+key,admin['token'],expected=404)
        stage = 'rejection and review permissions'
        review = '/workflow/review/'+owner['id']+'/'+key
        req('POST',review,owner['token'],expected=403,json={'decision':'accepted'})
        req('POST','/'+key+'/submit',owner['token'])
        queue = req('GET','/workflow/queue',admin['token'])
        assert any(r['correction_id']==key for r in queue['items'])
        req('POST',review,admin['token'],expected=422,json={'decision':'rejected'})
        rejected = req('POST',review,admin['token'],json={'decision':'rejected','reason':'Synthetic smoke reason'})
        assert rejected['status']=='rejected'
        assert req('GET','/'+key,owner['token'])['decisions'][-1]['reason']=='Synthetic smoke reason'
        stage = 'acceptance and immutable decisions'
        child = save(key,.3)['correction_id']
        req('POST','/'+child+'/submit',owner['token'])
        review2 = '/workflow/review/'+owner['id']+'/'+child
        accepted = req('POST',review2,admin['token'],json={'decision':'accepted'})
        assert req('POST',review2,admin['token'],json={'decision':'accepted'}) == accepted
        req('POST',review2,admin['token'],expected=409,json={'decision':'rejected','reason':'conflict'})
        stage = 'concurrent child revisions'
        def competing(x):
            try: return save(child,x)
            except AssertionError as e:
                assert str(e)=='Unexpected API status 409'
                return None
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(pool.map(competing,[.4,.45]))
        committed = [r for r in results if r]
        assert len(committed)==1 and committed[0]['review_status']=='draft'
        assert req('GET','/'+child,owner['token'])['eligible_for_training'] is False
        print('PASS: HTTPS auth, legacy PNG, state restoration, retry, ownership, moderation, decision retry/conflict, concurrent revision conflict, accepted-to-draft; no training.')
    except Exception:
        print('FAIL at stage: '+stage, file=sys.stderr)
        raise RuntimeError('Smoke test failed (details suppressed)') from None
    finally:
        # Every destructive operation is scoped to IDs returned by our own new
        # registrations; an extra email predicate guards account deletion.
        clean = True
        for a in accounts:
            try:
                prefix = 'v4-corrections/'+a['id']+'/'
                r=requests.post(url+'/storage/v1/object/list/'+quote(bucket,safe=''),headers=service,
                    json={'prefix':prefix,'limit':100},timeout=30)
                assert r.status_code==200
                names=[prefix+x['name'] for x in r.json() if x.get('id')]
                assert len(names)<100
                if names:
                    r=requests.delete(url+'/storage/v1/object/'+quote(bucket,safe=''),headers=service,json={'prefixes':names},timeout=30)
                    assert r.status_code==200
                db('DELETE','contour_revisions',params={'owner_id':'eq.'+a['id']})
                db('DELETE','auth_sessions',params={'user_id':'eq.'+a['id']})
                db('DELETE','users',params={'id':'eq.'+a['id'],'email':'eq.'+a['email']})
            except Exception:
                clean=False
        print('Synthetic fixture cleanup: '+('complete' if clean else 'FAILED; operator review required'))
        if not clean: raise RuntimeError('Synthetic fixture cleanup incomplete')


if __name__ == '__main__':
    try: main()
    except Exception: sys.exit(1)
