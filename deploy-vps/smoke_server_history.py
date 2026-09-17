"""Explicit synthetic-account smoke. No real users/contours are read or changed.
Run in candidate/container with REPORT_SMOKE_BASE for staging or default HTTPS.
Does not print credentials, IDs or response bodies.
"""
import concurrent.futures
import hashlib
import io
import json
import os
import secrets
import uuid
from urllib.parse import quote
import requests
from PIL import Image
from arborscan_v4.corrections_api import _config


def main():
    url,service,bucket=_config()
    base=os.getenv('REPORT_SMOKE_BASE','https://31.57.170.88/api/v4')
    accounts=[]; blobs={}; revisions=[]; stage='registration'
    def req(method,path,token=None,status=200,**kw):
        r=requests.request(method,base+path,headers={'Authorization':'Bearer '+token} if token else {},timeout=90,**kw)
        assert r.status_code==status, 'Unexpected status '+str(r.status_code)
        return r.json()
    try:
        for _ in range(2):
            email='history-smoke-'+uuid.uuid4().hex+'@example.invalid'
            r=requests.post('https://31.57.170.88/api/v3/auth/register',json={
                'name':'Server history synthetic fixture','email':email,'password':secrets.token_urlsafe(32)},timeout=60)
            assert r.status_code==200
            a=r.json(); assert a['user']['email']==email
            accounts.append({'id':a['user']['id'],'token':a['token'],'email':email})
        owner,other=accounts
        req('GET','/v4/reports',status=401)
        assert req('GET','/v4/reports/capabilities',owner['token'])['history_version']==1
        raw=io.BytesIO(); photo=Image.new('RGB',(200,100),'white'); exif=photo.getexif();exif[274]=6
        photo.save(raw,'JPEG',exif=exif);image=raw.getvalue()
        line=lambda a,b:[{'x':a[0],'y':a[1]},{'x':b[0],'y':b[1]}]
        snapshot={'version':1,'kind':'reference','change_source':'reference','captured_at':'2026-09-17T00:00:00Z',
          'report':{},'reference':{'version':1,'method':'known_object_segment_v1','coordinates':'normalized_oriented_image',
          'width':100,'height':200,'same_plane':True,'length_m':1,'reference':line((.1,.1),(.1,.3)),
          'tree':line((.5,.1),(.5,.9)),'crown':line((.2,.1),(.8,.1)),
          'outline':[{'x':.1,'y':.1},{'x':.1,'y':.3},{'x':.2,'y':.3}]},
          'environment':{'weather':{'value':{'temperature_c':8},'source':'synthetic_fixture','retrieved_at':'2026-09-16T00:00:00Z'}}}
        analysis=str(uuid.uuid4());first=str(uuid.uuid4())
        def save(version,parent=None,length=1,status=200):
            s=json.loads(json.dumps(snapshot));s['reference']['length_m']=length
            data={'analysis_id':analysis,'version_id':version,'snapshot':json.dumps(s)}
            if parent:data['parent_id']=parent
            result=req('POST','/v4/reports',owner['token'],status,data=data,files={'image':('fixture.jpg',image)})
            if status==200:
                row=result['record'];blobs[row['payload_sha256']]=owner['id']
                if version not in revisions:revisions.append(version)
                assert result['saved'] is True and result['persisted'] is True
            return result
        stage='roundtrip, retry, EXIF, ownership'
        a=save(first);assert save(first)==a
        r=req('GET','/v4/reports/'+first,owner['token'])
        assert r['snapshot']['image']['width']==100 and r['snapshot']['image']['height']==200
        assert abs(r['snapshot']['report']['height_m']-4)<1e-8
        assert r['snapshot']['environment']==snapshot['environment']
        req('GET','/v4/reports/'+first,other['token'],404)
        assert req('GET','/v4/reports',other['token'])['items']==[]
        save(first,length=2,status=409)
        stage='concurrent revisions'
        versions=[str(uuid.uuid4()),str(uuid.uuid4())]
        def compete(v):
            try:return save(v,first,2)
            except AssertionError as e:
                assert str(e)=='Unexpected status 409';return None
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            children=list(pool.map(compete,versions))
        assert sum(c is not None for c in children)==1
        assert req('GET','/v4/reports/'+first,owner['token'])['snapshot']['report']['height_m']==r['snapshot']['report']['height_m']
        assert len(req('GET','/v4/reports',owner['token'])['items'])==2
        stage='private object access'
        digest=a['record']['payload_sha256']
        object_url=url+'/storage/v1/object/public/'+quote(bucket,safe='')+'/report-versions/'+owner['id']+'/'+digest+'.json'
        assert requests.get(object_url,timeout=30).status_code!=200
        print('PASS: authorized save/read, immutable retry, original EXIF, historical environment, owner isolation, concurrent conflict, private files')
    finally:
        # Delete only this run's synthetic records, children before their parents.
        # Orphan blobs from deliberately conflicting submissions are also scoped
        # strictly to newly created fixture owners.
        for a in accounts:
            for version in reversed(revisions if a==accounts[0] else []):
                r=requests.delete(url+'/rest/v1/report_versions',headers=service,
                    params={'owner_id':'eq.'+a['id'],'version_id':'eq.'+version},timeout=30)
                assert r.status_code<300,'Fixture metadata cleanup failed'
            prefix='report-versions/'+a['id']
            r=requests.post(url+'/storage/v1/object/list/'+bucket,headers=service,json={'prefix':prefix,'limit':100},timeout=30)
            r.raise_for_status()
            for obj in r.json():
                name=obj['name'];assert '/' not in name and name.endswith('.json')
                r=requests.delete(url+'/storage/v1/object/'+bucket,headers=service,json={'prefixes':[prefix+'/'+name]},timeout=30)
                r.raise_for_status()
            for table,params in [('auth_sessions',{'user_id':'eq.'+a['id']}),('users',{'id':'eq.'+a['id'],'email':'eq.'+a['email']})]:
                r=requests.delete(url+'/rest/v1/'+table,headers=service,params=params,timeout=30)
                assert r.status_code<300,'Fixture account cleanup failed'
        print('Synthetic fixtures removed')


if __name__=='__main__':
    try:main()
    except Exception:
        print('Server history smoke failed; no credentials or response bodies printed')
        raise SystemExit(1)
