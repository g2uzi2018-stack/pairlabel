import io,json
from PIL import Image
from fastapi.testclient import TestClient
from app import db
from app.main import app,job_config
from app.config import load_config
from app.prompts import build_detection_prompt


def options():
 return {'annotation_profile':'general','classes':[{'id':0,'name':'gear','label':'齿轮'},{'id':1,'name':'bolt','label':'螺栓'}],
         'annotation_instructions':'只标注托盘里的完整物体，框不要包含投影。','max_attempts':1,'segmentation':True}


def test_prompt_preview_uses_actual_builder_without_creating_or_starting_job():
 client=TestClient(app);opts=options();r=client.post('/api/prompt-preview',json=opts)
 assert r.status_code==200
 data=r.json();assert data['annotation_profile']=='general'
 assert data['prompt']==build_detection_prompt(job_config(load_config(),opts),0)
 assert '齿轮' in data['prompt'] and opts['annotation_instructions'] in data['prompt']
 assert '炮/砲' not in data['prompt'] and '棋局' not in data['prompt']
 assert db.one('SELECT count(*) n FROM jobs')['n']==0


def test_uploaded_job_keeps_custom_categories_and_requirements_in_snapshot():
 client=TestClient(app);before=client.get('/api/config').json();opts=options()
 buf=io.BytesIO();Image.new('RGB',(32,24),'silver').save(buf,format='PNG')
 r=client.post('/api/jobs',data={'name':'通用标注快照回归','options':json.dumps(opts)},files=[('files',('托盘.png',buf.getvalue(),'image/png'))])
 assert r.status_code==200
 jid=r.json()['job']['id'];saved=db.job(jid)['config']
 assert saved['classes']==opts['classes'] and saved['annotation_instructions']==opts['annotation_instructions'] and saved['annotation_profile']=='general'
 opts['classes'][0]['label']='改名后的齿轮';opts['annotation_instructions']='已变更'
 assert db.job(jid)['config']==saved
 assert client.get('/api/config').json()==before
 assert db.one('SELECT count(*) n FROM items WHERE job_id=?',(jid,))['n']==1


def test_invalid_options_fail_before_image_import_and_preset_metadata_is_retained():
 client=TestClient(app);presets=client.get('/api/presets').json()
 chess={'annotation_profile':'xiangqi','classes':presets['xiangqi']['classes'],'annotation_instructions':'不要标注棋盘图案。'}
 r=client.post('/api/prompt-preview',json=chess);assert r.status_code==200 and chess['annotation_instructions'] in r.json()['prompt']
 invalid=[{'annotation_profile':'unknown'}, {'annotation_instructions':42}, {'classes':'杯子'}, {'classes':[{'id':0,'name':'a','label':'A'},{'id':1,'name':'a','label':'B'}]}, {'annotation_profile':'xiangqi','classes':options()['classes']}, {'runtime':{'port':1234}}]
 for opts in invalid:
  r=client.post('/api/prompt-preview',json=opts);assert r.status_code==400,(opts,r.text)
 r=client.post('/api/jobs',data={'directory':'/does/not/exist','options':'[]'});assert r.status_code==400 and 'JSON' in r.json()['detail']
 assert db.one('SELECT count(*) n FROM jobs')['n']==0


def test_category_names_roundtrip_without_ambiguous_text_delimiters():
 client=TestClient(app)
 for key in ['name','label']:
  for value in ['nuts|bolts','两行\n名称','回车\r名称',' 首尾空白 ']:
   opts=options();opts['classes'][0][key]=value
   r=client.post('/api/prompt-preview',json=opts)
   assert r.status_code==400 and '竖线' in r.json()['detail']
 # Renaming both display and export names retains the supplied chess metadata.
 presets=client.get('/api/presets').json();classes=presets['xiangqi']['classes']
 classes[0].update(name='custom_red_king',label='自定义红帅')
 r=client.post('/api/prompt-preview',json={'annotation_profile':'xiangqi','classes':classes})
 assert r.status_code==200 and 'custom_red_king' in r.json()['prompt']
