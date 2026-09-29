"""HTTP recovery smoke, run INSIDE isolated api container. Never target production."""
import base64,datetime,hashlib,io,json,os,secrets,time,uuid
from pathlib import Path
import requests
from PIL import Image,ImageDraw

assert os.environ['SUPABASE_URL']=='http://gateway:8000'
BASE='http://gateway:8000';key=os.environ['SUPABASE_SERVICE_KEY']
HEAD={'Authorization':'Bearer '+key,'apikey':key}
state_path=Path('/app/model-quality/recovery-fixture.private.json')
stage='start'
def db(method,path,**kw):
 r=requests.request(method,BASE+'/rest/v1/'+path,headers=HEAD,timeout=30,**kw)
 assert r.status_code<300,'db_status_'+str(r.status_code)
 return r.json() if r.content else None
def api(method,path,token=None,status=200,**kw):
 r=requests.request(method,BASE+path,headers={'Authorization':'Bearer '+token} if token else {},timeout=120,**kw)
 assert r.status_code==status,'http_status_'+str(r.status_code)
 return r.json() if r.content else None
try:
 if state_path.exists():
  s=json.loads(state_path.read_text())
  stage='restart_read'
  a=api('POST','/api/v3/auth/login',json=s['login'])
  for v in s['versions']:
   r=api('GET','/api/v4/v4/reports/'+v,a['token']);assert r['persisted'] is True
  r=api('GET','/api/v4/v4/corrections/'+s['accepted'],a['token']);assert r['review_status']=='accepted'
  r=api('GET','/api/v4/v4/corrections/'+s['draft'],a['token']);assert r['review_status']=='draft'
  j=db('GET','ml_jobs',params={'id':'eq.'+s['job']})[0];assert j['state']=='failed'
  job_dir=Path('/app/model-quality/jobs')/s['job']
  assert json.loads((job_dir/'stage.json').read_text())['stage']=='snapshot_verification'
  assert not (job_dir/'candidate.pt').exists() and not (job_dir/'runs').exists()
  assert db('GET','ml_active_models')==s['active_models']
  print(json.dumps({'restart_login_reports_contours_worker':'passed','versions':2,'training_started':False}));raise SystemExit(0)
 stage='health_auth'
 for v in ('v3','v4'):assert api('GET','/api/'+v+'/health')['status']=='ok'
 api('GET','/api/v4/v4/reports',status=401)
 accounts=[]
 for _ in range(2):
  login={'email':'recovery-'+uuid.uuid4().hex+'@invalid.test','password':secrets.token_urlsafe(32)}
  a=api('POST','/api/v3/auth/register',json={'name':'AS14 recovery fixture',**login});a['login']=login;accounts.append(a)
 owner,admin=accounts
 api('POST','/api/v3/auth/login',json={**owner['login'],'password':'wrong'},status=401)
 assert api('POST','/api/v3/auth/login',json=owner['login'])['user']['id']==owner['user']['id']
 db('PATCH','users',params={'id':'eq.'+admin['user']['id']},json={'role':'admin'})
 stage='restored_read'
 tokens={};cutoff={'created_at':'lt.2026-09-28T13:30:59Z'}
 reports=db('GET','report_versions',params=cutoff);contours=db('GET','contour_revisions',params=cutoff)
 for uid in {r['owner_id'] for r in reports+contours}:
  token=secrets.token_urlsafe(32);tokens[uid]=token
  db('POST','auth_sessions',json={'token':token,'user_id':uid,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expires_at':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat()})
 photo=None
 for r in reports:
  content=api('GET','/api/v4/v4/reports/'+r['version_id'],tokens[r['owner_id']])['snapshot']
  raw=base64.b64decode(content['image']['original_base64']);assert hashlib.sha256(raw).hexdigest()==r['image_sha256'];photo=raw
  api('GET','/api/v4/v4/reports/'+r['version_id'],owner['token'],status=404)
 for r in contours:
  content=api('GET','/api/v4/v4/corrections/'+r['correction_id'],tokens[r['owner_id']])
  for field,digest in [('original_image_base64','image_sha256'),('mask_png_base64','mask_sha256')]:
   raw=base64.b64decode(content[field]);assert hashlib.sha256(raw).hexdigest()==content[digest]
 stage='inference'
 im=Image.open(io.BytesIO(photo)).convert('RGB');im.thumbnail((640,640));out=io.BytesIO();im.save(out,'JPEG')
 inferred=api('POST','/api/v4/v4/analyze-tree',owner['token'],files={'file':('bounded.jpg',out.getvalue(),'image/jpeg')})
 stage='new_versions'
 image=io.BytesIO();Image.new('RGB',(48,64),'gray').save(image,'PNG');raw=image.getvalue()
 mask=io.BytesIO();pixels=Image.new('L',(48,64),0);ImageDraw.Draw(pixels).rectangle((10,6,30,54),fill=255);pixels.save(mask,'PNG')
 line=lambda x,y:[{'x':x[0],'y':x[1]},{'x':y[0],'y':y[1]}]
 snap={'version':1,'kind':'reference','change_source':'reference','captured_at':'2026-09-28T00:00:00Z','report':{},
  'reference':{'version':1,'method':'known_object_segment_v1','coordinates':'normalized_oriented_image','width':48,'height':64,'same_plane':True,'length_m':1,
    'reference':line((.1,.1),(.1,.3)),'tree':line((.5,.1),(.5,.9)),'crown':line((.2,.1),(.8,.1)),
    'outline':[{'x':.1,'y':.1},{'x':.1,'y':.3},{'x':.2,'y':.3}]}}
 aid=str(uuid.uuid4());versions=[str(uuid.uuid4()),str(uuid.uuid4())]
 for index,v in enumerate(versions):
  data={'analysis_id':aid,'version_id':v,'snapshot':json.dumps(snap)}
  if index:data['parent_id']=versions[0]
  result=api('POST','/api/v4/v4/reports',owner['token'],data=data,files={'image':('fixture.png',raw)})
  assert result['saved'] is True
 stage='moderation'
 def save(parent=None,x=.2):
  editor={'version':1,'coordinates':'normalized_oriented_image','width':48,'height':64,'closed':True,'points':[{'x':x,'y':.1},{'x':.8,'y':.1},{'x':.5,'y':.8}]}
  data={'analysis_id':aid,'editor_state':json.dumps(editor)}
  if parent:data['parent_id']=parent
  r=api('POST','/api/v4/v4/corrections/workflow',owner['token'],data=data,files={'image':('photo.png',raw),'mask':('mask.png',mask.getvalue())});assert r['saved'];return r['correction_id']
 first=save();assert save()==first
 api('GET','/api/v4/v4/corrections/workflow/queue',owner['token'],status=403)
 def review(cid,decision,reason=None):
  api('POST','/api/v4/v4/corrections/'+cid+'/submit',owner['token'])
  r=api('POST','/api/v4/v4/corrections/workflow/review/'+owner['user']['id']+'/'+cid,admin['token'],json={'decision':decision,**({'reason':reason} if reason else {})});assert r['status']==decision
 review(first,'rejected','Recovery fixture rejection')
 accepted=save(first,.3);review(accepted,'accepted');draft=save(accepted,.4)
 stage='worker_fixture'
 active=db('GET','ml_active_models');snapshot=str(uuid.uuid4());job=str(uuid.uuid4())
 db('POST','rpc/ml_transition',json={'p_action':'snapshot','p_id':snapshot,'p_actor':admin['user']['id'],'p_data':{'model_type':'segmentation','manifest':{'archive_sha256':'0'*64}}})
 db('POST','rpc/ml_transition',json={'p_action':'enqueue','p_id':job,'p_actor':admin['user']['id'],'p_data':{'snapshot_id':snapshot,'params':{'max_seconds':30}}})
 state_path.write_text(json.dumps({'login':owner['login'],'versions':versions,'accepted':accepted,'draft':draft,'job':job,'active_models':active}))
 state_path.chmod(0o600)
 print(json.dumps({'auth_owners':'passed','restored_reports':len(reports),'restored_contours':len(contours),'photo_mask_hashes':'passed','bounded_inference':'passed','new_versions':2,'moderation':'passed','worker_fixture':'queued_invalid_archive_no_training'}))
except Exception as e:
 print(json.dumps({'failed_stage':stage,'error_type':type(e).__name__,'status':str(e) if str(e).startswith(('http_status_','db_status_')) else None}));raise SystemExit(1)
