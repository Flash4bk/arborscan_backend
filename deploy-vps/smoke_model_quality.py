"""HTTP integration with isolated synthetic accounts. Never trains or activates.

Run inside candidate runtime. MODEL_QUALITY_SMOKE_BASE selects staging or HTTPS.
Does not print credentials, private records, or response bodies.
"""
import os
import json
import secrets
import uuid
import cv2
import numpy as np
import requests
from arborscan_v4.corrections_api import _config


def main():
    url,service,bucket=_config()
    base=os.getenv('MODEL_QUALITY_SMOKE_BASE','https://31.57.170.88/api/v4')
    accounts=[];snapshots=[];labels=[];archives=[]
    def db(method,table,**kwargs):
        r=requests.request(method,url+'/rest/v1/'+table,headers=service,timeout=30,**kwargs)
        assert r.status_code<300,'Fixture database operation failed'
        return r
    def req(method,path,token=None,status=200,**kwargs):
        r=requests.request(method,base+path,headers={'Authorization':'Bearer '+token} if token else {},timeout=120,**kwargs)
        assert r.status_code==status,'Unexpected HTTP '+str(r.status_code)+' for '+path.split('/')[2]
        return r.json()
    try:
        for _ in range(2):
            email='model-quality-smoke-'+uuid.uuid4().hex+'@example.invalid'
            r=requests.post('https://31.57.170.88/api/v3/auth/register',json={'name':'ML synthetic fixture','email':email,'password':secrets.token_urlsafe(32)},timeout=60)
            assert r.status_code==200
            a=r.json();assert a['user']['email']==email
            accounts.append({'id':a['user']['id'],'token':a['token'],'email':email})
        owner,admin=accounts
        prefix='/v4/model-quality'
        req('GET',prefix+'/status',status=401)
        req('GET',prefix+'/data',owner['token'],403)
        req('POST',prefix+'/activate',owner['token'],403,json={'model_id':None,'expected_generation':0})
        db('PATCH','users',params={'id':'eq.'+admin['id'],'email':'eq.'+admin['email']},json={'role':'admin'})
        assert req('GET',prefix+'/status',admin['token'])['version']==1
        image=cv2.imencode('.png',np.full((64,48,3),140,np.uint8))[1].tobytes()
        mask=np.zeros((64,48),np.uint8);mask[4:60,8:40]=255
        mask=cv2.imencode('.png',mask)[1].tobytes()
        state={'version':1,'coordinates':'normalized_oriented_image','width':48,'height':64,'closed':True,
               'points':[{'x':.2,'y':.1},{'x':.8,'y':.1},{'x':.5,'y':.9}]}
        result=req('POST','/v4/corrections/workflow',owner['token'],data={'analysis_id':str(uuid.uuid4()),'editor_state':json.dumps(state)},files={'image':('fixture.png',image),'mask':('mask.png',mask)})
        correction=result['correction_id']
        selection=[{'owner_id':owner['id'],'correction_id':correction}]
        key=str(uuid.uuid4());labels.append(key)
        body={'operation_id':key,'owner_id':owner['id'],'correction_id':correction,'scientific_name':'Pinus sylvestris',
              'rank':'species','authority':'manual_reference','taxon_id':'synthetic-'+uuid.uuid4().hex,
              'evidence':'Synthetic transport fixture; not botanical evidence','confirmed':True}
        req('POST',prefix+'/labels',owner['token'],403,json=body)
        label=req('POST',prefix+'/labels',admin['token'],json=body)
        assert req('POST',prefix+'/labels',admin['token'],json=body)==label
        assert req('GET',prefix+'/labels/'+owner['id']+'/'+correction,admin['token'])['items'][0]['id']==key
        # Independent label does not accept the contour.
        assert req('GET','/v4/corrections/'+correction,owner['token'])['review_status']!='accepted'
        for kind in ('segmentation','classification'):
            sid=str(uuid.uuid4());snapshots.append(sid)
            body={'operation_id':sid,'model_type':kind,'selection':selection}
            snap=req('POST',prefix+'/snapshots',admin['token'],json=body)
            archives.append(snap['manifest']['archive_sha256'])
            assert req('POST',prefix+'/snapshots',admin['token'],json=body)==snap
            assert len(snap['manifest']['items'])==(1 if kind=='classification' else 0)
            assert snap['manifest']['training_ready'] is False
            req('POST',prefix+'/jobs',admin['token'],422,json={'operation_id':str(uuid.uuid4()),'snapshot_id':sid})
        print('PASS: HTTPS/app auth, admin-only operations, independent taxon label, immutable snapshot retries, insufficient-data gate. No training or activation.')
    finally:
        for sid in snapshots:db('DELETE','ml_snapshots',params={'id':'eq.'+sid})
        for digest in archives:
            remaining=db('GET','ml_snapshots',params={'manifest->>archive_sha256':'eq.'+digest,'limit':'1'}).json()
            if not remaining:
                r=requests.delete(url+'/storage/v1/object/'+bucket,headers=service,json={'prefixes':['model-quality/'+digest]},timeout=30);r.raise_for_status()
        for key in reversed(labels):db('DELETE','ml_taxon_labels',params={'id':'eq.'+key})
        for a in accounts:
            db('DELETE','contour_revisions',params={'owner_id':'eq.'+a['id']})
            prefix='v4-corrections/'+a['id']
            r=requests.post(url+'/storage/v1/object/list/'+bucket,headers=service,json={'prefix':prefix,'limit':100},timeout=30);r.raise_for_status()
            for obj in r.json():
                assert '/' not in obj['name']
                r=requests.delete(url+'/storage/v1/object/'+bucket,headers=service,json={'prefixes':[prefix+'/'+obj['name']]},timeout=30);r.raise_for_status()
            db('DELETE','auth_sessions',params={'user_id':'eq.'+a['id']})
            db('DELETE','users',params={'id':'eq.'+a['id'],'email':'eq.'+a['email']})
        print('Synthetic fixture cleanup complete')


if __name__=='__main__':
    try:main()
    except Exception:
        print('ML integration smoke failed; no secrets or private response bodies printed')
        raise SystemExit(1)
