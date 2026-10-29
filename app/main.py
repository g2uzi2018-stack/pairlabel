import asyncio,contextlib,copy,hashlib,io,json,os,sqlite3,time,uuid
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError
from fastapi import FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,StrictInt
from . import db
from .annotation_formats import format_preview
from .job_access import job_guard
from .config import ROOT,DATA,CLASS_PRESETS,load_config,save_config,readiness,validate
from .profiles import is_xiangqi
from .prompts import build_detection_prompt
from .exports import build_export,export_busy
from .geometry import validate_prediction
from .pipeline import Worker,preview
from .review import FAILED_SQL,recognition_failed,relabel,review_revision,visual_edit,agent_classify
from .classification import unreadable

worker=Worker()
@contextlib.asynccontextmanager
async def lifespan(app):
 db.init();(ROOT/'logs').mkdir(exist_ok=True);worker.start()
 yield
 worker.stop()
app=FastAPI(title='PairLabel 协标 · 本地图片自动标注',lifespan=lifespan)
from starlette.middleware.trustedhost import TrustedHostMiddleware
bind_host=os.environ.get('LABELER_HOST','127.0.0.1')
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver',bind_host])

@app.middleware('http')
async def local_guard(request:Request,call_next):
 # Keep browser writes same-origin on both loopback and explicitly bound LAN addresses.
 if request.method in ['POST','PUT','PATCH','DELETE']:
  origin=request.headers.get('origin')
  if origin:
   from urllib.parse import urlsplit
   if urlsplit(origin).netloc!=request.headers.get('host'):return JSONResponse({'detail':'不允许跨站写入'},status_code=403)
 response=await call_next(request)
 # Always fetch the current UI shell; asset URLs carry content hashes.
 if request.url.path in ('/','/index.html'):response.headers['Cache-Control']='no-store'
 return response

@app.exception_handler(ValueError)
async def value_error(request,exc):return JSONResponse({'detail':str(exc)},status_code=400)

@app.get('/api/health')
def health():return {'status':'ok','local_only':bind_host in ['localhost','127.0.0.1','::1'],'listen_host':bind_host}
@app.get('/api/status')
def status():
 c=load_config();mem={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.split(':')[0] in ['MemTotal','MemAvailable']:mem[line.split(':')[0]]=int(line.split()[1])
 active=db.one("SELECT id FROM jobs WHERE status IN ('running','pausing') AND deleted_at=0 ORDER BY created LIMIT 1")
 return {'models':readiness(c),'runtime':worker.manager.state,'active_job_id':active['id'] if active else None,'memory_kb':mem,'project':str(ROOT),'download_log':(ROOT/'logs/download.log').read_text(errors='replace')[-1800:] if (ROOT/'logs/download.log').exists() else ''}
@app.get('/api/config')
def get_config():return load_config()
@app.get('/api/presets')
def get_presets():
 return {key:{'label':'象棋增强' if key=='xiangqi' else '通用物件','classes':copy.deepcopy(value)} for key,value in CLASS_PRESETS.items()}

def job_config(base,options):
 if not isinstance(options,dict):raise ValueError('任务选项必须是 JSON 对象')
 allowed={'segmentation','circle_audit','allow_empty','max_attempts','annotation_profile','annotation_instructions','classes'}
 if set(options)-allowed:raise ValueError('不支持的任务选项：'+', '.join(sorted(set(options)-allowed)))
 c=copy.deepcopy(base)
 for key in ['segmentation','circle_audit','allow_empty']:
  if key in options:
   if type(options[key]) is not bool:raise ValueError('选项必须是布尔值')
   c[key]=options[key]
 if 'max_attempts' in options:
  if type(options['max_attempts']) is not int or not 1<=options['max_attempts']<=4:raise ValueError('尝试次数应为 1–4')
  c['max_attempts']=options['max_attempts']
 if 'classes' in options:
  cs=options['classes']
  if not isinstance(cs,list) or not 1<=len(cs)<=200 or any(not isinstance(x,dict) for x in cs):raise ValueError('类别应为 1–200 项的对象数组')
  c['classes']=copy.deepcopy(cs)
 for key in ['annotation_profile','annotation_instructions']:
  if key in options:c[key]=options[key]
 validate(c)
 if is_xiangqi(c) and not all(x.get('piece') and x.get('color') in ['red','black'] for x in c['classes']):raise ValueError('象棋增强类别需要保留棋子元数据；请载入象棋预设，或切换通用模式自定义物件')
 return c

@app.post('/api/prompt-preview')
def preview_prompt(options:dict):
 c=job_config(load_config(),options)
 return {'annotation_profile':'xiangqi' if is_xiangqi(c) else 'general','classes':c['classes'],'prompt':build_detection_prompt(c,0),'independence':'模型 A 和 B 独立使用同一标注规范，互不接收对方结果'}

@app.put('/api/config')
def put_config(c:dict):
 checked=job_config(c,{})
 save_config(checked);return {'ok':True,'note':'仅影响新任务，已有任务保持配置快照'}
@app.get('/api/jobs')
def jobs():return [job for r in db.rows('SELECT id FROM jobs WHERE deleted_at=0 ORDER BY created DESC LIMIT 100') if (job:=db.job(r['id']))]

def require_job(jid):
 j=db.job(jid)
 if not j:raise HTTPException(404,'任务不存在')
 return j

def require_item(iid):
 i=db.item(iid)
 if not i:raise HTTPException(404,'图片不存在')
 require_job(i['job_id'])
 return i

def import_image(jid,raw,name):
 if len(raw)>80*1024*1024:raise ValueError('单张图片不能超过 80MB')
 with Image.open(io.BytesIO(raw)) as original:
  if original.width*original.height>36_000_000:raise ValueError('单图不能超过 3600 万像素')
  if getattr(original,'n_frames',1)>1:raise ValueError('请先将多帧图片拆为静态图片')
  im=ImageOps.exif_transpose(original).convert('RGB');w,h=im.size
  # Include dimensions in content hash, not compressed-file metadata.
  digest=hashlib.sha256(f'{w}x{h}:'.encode()+im.tobytes()).hexdigest()
  iid=uuid.uuid4().hex;rel=Path('jobs')/jid/'images'/f'{iid}.png';target=DATA/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if db.one('SELECT id FROM items WHERE job_id=? AND hash=?',(jid,digest)):return False
  im.save(target,format='PNG')
  thumb=im.copy();thumb.thumbnail((240,240));thumb.save(target.with_suffix('.thumb.jpg'),format='JPEG',quality=85)
  view=im.copy();view.thumbnail((1800,1800));view.save(target.with_suffix('.view.jpg'),format='JPEG',quality=92)
  try:db.execute('INSERT INTO items(id,job_id,name,hash,path,width,height) VALUES(?,?,?,?,?,?,?)',(iid,jid,name[:500],digest,str(rel),w,h))
  except Exception:target.unlink(missing_ok=True);raise
 return True

@app.post('/api/jobs')
async def create_job(name:str=Form('图片标注任务'),directory:str=Form(''),options:str=Form('{}'),files:list[UploadFile]=File(default=[])):
 c=job_config(load_config(),json.loads(options))
 if not directory.strip() and not files:raise ValueError('请选择图片或填写设备上的目录')
 paths=[]
 if directory.strip():
  root=Path(directory).expanduser().resolve();allowed=[Path(p).expanduser().resolve() for p in c['allowed_roots']]
  if not root.is_dir():raise ValueError('目录不存在，路径应是 Jetson 上的目录')
  if not any(root.is_relative_to(p) for p in allowed):raise ValueError('目录不在允许读取的范围；可在设置中添加 allowed_roots')
  if root==DATA or root.is_relative_to(DATA):raise ValueError('不能将任务数据目录作为输入')
  for p in sorted(root.rglob('*')):
   if p.suffix.lower() not in ['.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff']:continue
   resolved=p.resolve()
   if not resolved.is_relative_to(root) or resolved.is_relative_to(DATA):continue
   if p.is_file():paths.append((p,str(p.relative_to(root))))
   if len(paths)>5000:raise ValueError('每任务最多 5000 张图片，请拆分目录')
 if len(paths)+len(files)>5000:raise ValueError('每任务最多 5000 张图片')
 jid=db.new_job(name[:100],c);(DATA/'jobs'/jid/'results').mkdir(parents=True,exist_ok=True);errors=[];imported=0;duplicates=0
 try:
  for p,relative in paths:
   try:
    if p.stat().st_size>80*1024*1024:raise ValueError('文件超过 80MB')
    if await asyncio.to_thread(import_image,jid,p.read_bytes(),relative):imported+=1
    else:duplicates+=1
   except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError) as e:errors.append({'name':relative,'error':str(e)[:200]})
  for f in files:
   try:
    raw=await f.read(80*1024*1024+1)
    if await asyncio.to_thread(import_image,jid,raw,Path(f.filename or 'image').name):imported+=1
    else:duplicates+=1
   except (OSError,ValueError,UnidentifiedImageError,Image.DecompressionBombError) as e:errors.append({'name':f.filename,'error':str(e)[:200]})
   finally:await f.close()
  if not imported:raise ValueError('没有成功导入的图片：'+str(errors[:3]))
  db.audit(jid,None,'import',{'imported':imported,'duplicates':duplicates,'errors':errors})
  db.set_job(jid,'queued');worker.wake.set();return {'job':db.job(jid),'imported':imported,'duplicates':duplicates,'errors':errors}
 except Exception as e:db.set_job(jid,'failed',str(e));raise

@app.get('/api/trash')
def trash_jobs():return [job for r in db.rows('SELECT id FROM jobs WHERE deleted_at>0 ORDER BY deleted_at DESC') if (job:=db.job(r['id'],include_deleted=True)) and job['deleted_at']]

@app.delete('/api/jobs/{jid}')
def trash_job(jid:str):
 with job_guard(jid):
  job=require_job(jid)
  if job['status'] in ['running','queued','pausing','importing']:raise HTTPException(409,'请先暂停任务，等待当前识别保存后再删除')
  if export_busy(jid):raise HTTPException(409,'正在导出该任务，请完成后再删除')
  now=time.time()
  with db.connection() as con:
   con.execute('UPDATE jobs SET deleted_at=?,updated=? WHERE id=?',(now,now,jid))
   con.execute('INSERT INTO audit(job_id,kind,detail,created) VALUES(?,?,?,?)',(jid,'trash',db.dumps({'name':job['name']}),now))
  return {'ok':True,'id':jid,'message':'任务已移入回收站，图片和标注保留，可恢复；原始采集目录不受影响。'}

@app.post('/api/trash/{jid}/restore')
def restore_job(jid:str):
 with job_guard(jid):
  job=db.job(jid,include_deleted=True)
  if not job or not job['deleted_at']:raise HTTPException(404,'回收站中没有此任务')
  now=time.time()
  with db.connection() as con:
   con.execute('UPDATE jobs SET deleted_at=0,updated=? WHERE id=?',(now,jid))
   con.execute('INSERT INTO audit(job_id,kind,detail,created) VALUES(?,?,?,?)',(jid,'restore','{}',now))
  return require_job(jid)

@app.get('/api/items/{iid}/format-preview')
def get_format_preview(iid:str,format:str='yolo',include_uncertain:bool=False):
 i=require_item(iid);job=require_job(i['job_id'])
 objects=(i.get('result') or {}).get('objects',[])
 result=format_preview(objects,i['width'],i['height'],format,job['config']['classes'])
 allowed=['accepted','approved']+(['uncertain'] if include_uncertain else [])
 reasons=[]
 if job['status'] in ['running','queued','pausing','importing']:reasons.append('任务正在处理，请先暂停或等待完成再导出。')
 if i['status'] not in allowed:reasons.append('当前图片不在导出范围：默认只含自动通过和人工确认；疑难需显式勾选包含。')
 if not result['format_eligible']:reasons.append(result['format_reason'])
 return {**result,'item_id':iid,'format':format,'included':not reasons,'reasons':reasons}

@app.get('/api/jobs/{jid}')
def get_job(jid:str):return require_job(jid)
@app.get('/api/jobs/{jid}/items')
def items(jid:str,status:str='all',offset:int=0,limit:int=100):
 require_job(jid);limit=max(1,min(limit,500));offset=max(0,offset)
 where='job_id=?';args=[jid]
 if status in ['failed','error']:where+=' AND '+FAILED_SQL
 elif status=='uncertain':where+=" AND status='uncertain' AND NOT "+FAILED_SQL
 elif status!='all':where+=' AND status=?';args.append(status)
 total=db.one(f'SELECT count(*) n FROM items WHERE {where}',args)['n']
 rs=db.rows(f'SELECT * FROM items WHERE {where} ORDER BY name,id LIMIT ? OFFSET ?',[*args,limit,offset])
 for r in rs:
  r.pop('path');r['result']=json.loads(r['result']) if r['result'] else None;r['issues']=json.loads(r['issues'])
  r['recognition_failed']=recognition_failed(r)
  r['review_revision']=review_revision(r)
 return {'items':rs,'total':total,'offset':offset}
class Action(BaseModel):
 action:str
 item_id:Optional[str]=None
@app.post('/api/jobs/{jid}/action')
def job_action(jid:str,body:Action):
 with job_guard(jid):
  job=require_job(jid)
  if export_busy(jid):raise ValueError('正在导出该任务，请稍后再修改任务状态')
  if body.action=='pause':
   if job['status'] in ['queued','running']:db.set_job(jid,'pausing' if job['status']=='running' else 'paused')
  elif body.action=='resume':
   if job['status'] not in ['running','queued']:
    if not readiness(job['config'])['ready']:raise ValueError('模型尚未下载完成，请稍后继续')
    db.set_job(jid,'queued');worker.wake.set()
  elif body.action in ['retry_uncertain','retry_failed']:
   if job['status'] in ['running','queued','pausing','importing']:raise ValueError('请先暂停当前任务')
   with db.connection() as con:
    condition=FAILED_SQL if body.action=='retry_failed' else "status IN ('uncertain','error')"
    query=f"SELECT id,attempt FROM items WHERE job_id=? AND {condition}";params=[jid]
    if body.item_id:
     if con.execute("SELECT count(*) FROM items WHERE job_id=? AND status='pending'",(jid,)).fetchone()[0]:raise ValueError('任务还有待处理图片，请先完成任务后再单独重试')
     query+=' AND id=?';params.append(body.item_id)
    selected=con.execute(query,params).fetchall()
    if not selected:raise ValueError('没有可重试的识别失败图片' if body.action=='retry_failed' else '没有可重试的疑难图片')
    for r in selected:
     # Preserve run history and increase the per-job attempt ceiling.
     next_attempt=r['attempt']+1;con.execute("UPDATE items SET status='pending',attempt=? WHERE id=?",(next_attempt,r['id']))
    maximum=con.execute("SELECT max(attempt) FROM items WHERE job_id=? AND status='pending'",(jid,)).fetchone()[0]
    if maximum is None:raise ValueError('没有可重试的疑难图片')
    cfg=job['config'];cfg['max_attempts']=max(cfg['max_attempts'],maximum+1);con.execute('UPDATE jobs SET config=? WHERE id=?',(db.dumps(cfg),jid))
   db.set_job(jid,'queued');worker.wake.set()
  else:raise ValueError('未知操作')
  db.audit(jid,None,'job_action',{'action':body.action});return require_job(jid)
@app.get('/api/items/{iid}')
def get_item(iid:str):
 i=require_item(iid);i.pop('path');i['runs']=db.rows('SELECT * FROM runs WHERE item_id=? ORDER BY attempt,role',(iid,))
 for r in i['runs']:
  for k in ['prediction','metrics']:r[k]=json.loads(r[k]) if r[k] else None
 i['audit']=db.rows('SELECT * FROM audit WHERE item_id=? ORDER BY id',(iid,));return i
@app.get('/api/items/{iid}/image')
def get_image(iid:str):return FileResponse(DATA/require_item(iid)['path'],media_type='image/png')
@app.get('/api/items/{iid}/thumbnail')
def get_thumbnail(iid:str):
 p=(DATA/require_item(iid)['path']).with_suffix('.thumb.jpg');return FileResponse(p if p.exists() else DATA/require_item(iid)['path'])
@app.get('/api/items/{iid}/view')
def get_view(iid:str):
 p=(DATA/require_item(iid)['path']).with_suffix('.view.jpg');return FileResponse(p if p.exists() else DATA/require_item(iid)['path'])
@app.get('/api/items/{iid}/crops/{attempt}/{role}/{page}')
def get_crop_evidence(iid:str,attempt:int,role:str,page:int,thumbnail:bool=False):
 i=require_item(iid)
 if role not in ['generator','reviewer']:raise HTTPException(404,'证据不存在')
 p=DATA/'jobs'/i['job_id']/'results'/f'{iid}.crops.attempt-{attempt}.{role}.page-{page}.png'
 if not p.is_file():raise HTTPException(404,'证据不存在')
 if thumbnail and p.with_suffix('.jpg').exists():return FileResponse(p.with_suffix('.jpg'),media_type='image/jpeg')
 return FileResponse(p,media_type='image/png')
@app.get('/api/items/{iid}/preview')
def get_preview(iid:str):
 i=require_item(iid);p=DATA/'jobs'/i['job_id']/'results'/f'{iid}.jpg'
 return FileResponse(p if p.exists() else DATA/i['path'])
class Review(BaseModel):
 action:str
 reason:str=Field(default='',max_length=2000)
 objects:Optional[list[dict]]=None
 class_ids:Optional[list[StrictInt]]=None
 expected_class_ids:Optional[list[StrictInt]]=None
 expected_revision:Optional[str]=None
@app.post('/api/items/{iid}/review')
def manual_review(iid:str,body:Review):
 i=require_item(iid);j=require_job(i['job_id'])
 with job_guard(j['id']):
  i=require_item(iid)
  j=require_job(i['job_id'])
  if export_busy(j['id']):raise ValueError('正在导出该任务，请稍后再修改标注')
  # Worker candidates are pending items only. Other completed images remain editable
  # during a retry; queueing and manual writes share job_guard to avoid claim races.
  if i['status']=='pending':raise ValueError('本图正在等待或进行识别，请完成后再编辑；其他已处理图片仍可复核')
  if body.expected_revision is not None and body.expected_revision!=i['review_revision']:raise ValueError('本图结果已更新，请重新打开图片后确认，旧修改未覆盖新结果')
  if body.action not in ['approve','reject','edit','labels','visual']:raise ValueError('未知复核操作')
  if body.action not in ['labels','visual'] and not body.reason.strip():raise ValueError('请填写复核依据')
  result=copy.deepcopy(i['result'] or {'objects':[]});label_changes=[];sources=None;reason=body.reason.strip()
  if body.action=='labels':
   result,label_changes,reason=relabel(result,body.class_ids,j['config']['classes'],body.expected_class_ids)
  if body.action=='visual':
   if body.expected_revision is None:raise ValueError('请重新打开图片后进行可视化编辑')
   result,sources,reason=visual_edit(i,body.objects,j['config']['classes'],i['width'],i['height'],lambda rid:db.one('SELECT * FROM runs WHERE id=?',(rid,)))
  if body.action=='edit':
   if body.objects is None:raise ValueError('需要 objects')
   new=validate_prediction({'objects':body.objects},j['config']['classes'],i['width'],i['height'],normalized=False)
   from .classification import assign_class
   for source,obj in zip(body.objects,new['objects']):
    if unreadable(source):assign_class(obj,-1,j['config']['classes'])
   result={**result,**new};result['human_edited']=True
  status='rejected' if body.action=='reject' else ('uncertain' if any(unreadable(o) for o in result['objects']) else 'approved')
  result['human_review']={'reason':reason,'action':body.action,'time':time.time(),**({'label_changes':label_changes} if body.action=='labels' else {}),**({'object_sources':sources} if sources is not None else {})}
  issues=[issue for issue in (i['issues'] or []) if issue.get('type')!='classification_unreadable']
  if any(unreadable(o) for o in result['objects']):
   issues=[issue for issue in issues if issue.get('type')!='inference_error']
   issues.append({'type':'classification_unreadable','message':'存在看不清／无法可靠分类的目标，暂缓训练导出'})
  db.execute('UPDATE items SET status=?,result=?,issues=? WHERE id=?',(status,db.dumps(result),db.dumps(issues),iid))
  db.audit(i['job_id'],iid,'human_review',{'action':body.action,'reason':reason,'before':i['result'],'after':result})
  p=DATA/'jobs'/i['job_id']/'results';preview(DATA/i['path'],result,j['config']['classes'],p/f'{iid}.jpg')
  (p/f'{iid}.human-{time.time_ns()}.json').write_text(json.dumps({'item':iid,'action':body.action,'reason':reason,'result':result},ensure_ascii=False,indent=2))
  return get_item(iid)
class ExportOptions(BaseModel):
 format:str='all'
 include_uncertain:bool=False
 val_ratio:float=.2

class ClassificationReview(BaseModel):
 class_ids:list[Optional[StrictInt]]
 observations:list[dict]
 expected_revision:str
 batch:str=Field(min_length=1,max_length=100)
 source_run_id:Optional[StrictInt]=None

@app.post('/api/items/{iid}/classification-review')
def classification_review(iid:str,body:ClassificationReview):
 i=require_item(iid);j=require_job(i['job_id'])
 with job_guard(j['id']):
  i=require_item(iid);j=require_job(i['job_id'])
  if export_busy(j['id']):raise ValueError('正在导出，请稍后保存类别核对')
  if (i['result'] or {}).get('classification_review',{}).get('batch')==body.batch:return get_item(iid)
  if i['status'] in ('approved','rejected') or (i['result'] or {}).get('human_review'):raise ValueError('保留已有人工结果，不自动覆盖')
  if i['status']=='pending':raise ValueError('本图正在识别，请完成后再核对')
  if i['review_revision']!=body.expected_revision:raise ValueError('本图结果已更新，类别核对未覆盖新结果')
  recovery=None
  if body.source_run_id is not None:
   run=db.one('SELECT * FROM runs WHERE id=? AND item_id=? AND status=?',(body.source_run_id,iid,'ok'))
   if not run:raise ValueError('轮廓来源不属于本图的成功模型记录')
   recovery=(json.loads(run['prediction']) or {}).get('objects',[])
   if not recovery or any(len(o.get('polygon',[]))<3 for o in recovery):raise ValueError('采用结果必须保留已有有效轮廓')
  result,issues,changes=agent_classify(i,body.class_ids,body.observations,j['config']['classes'],body.batch,recovery)
  status='uncertain' if issues else 'accepted'
  detail={'source':'codex_agent','model':'gpt-6-luna','reasoning_effort':'max','speed':'not_exposed','batch':body.batch,
          'before':i['result'],'before_issues':i['issues'],'after':result,'changes':changes,'source_run_id':body.source_run_id}
  with db.connection() as con:
   con.execute('UPDATE items SET status=?,result=?,issues=? WHERE id=?',(status,db.dumps(result),db.dumps(issues),iid))
   con.execute('INSERT INTO audit(job_id,item_id,kind,detail,created) VALUES(?,?,?,?,?)',(j['id'],iid,'agent_classification_review',db.dumps(detail),time.time()))
  p=DATA/'jobs'/j['id']/'results';preview(DATA/i['path'],result,j['config']['classes'],p/f'{iid}.jpg')
  (p/f'{iid}.agent-{time.time_ns()}.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
  return get_item(iid)

@app.post('/api/jobs/{jid}/export')
def prepare_export(jid:str,body:ExportOptions):
 p=build_export(jid,body.format,body.include_uncertain,body.val_ratio)
 return {'server_path':str(p),'download_url':f'/api/exports/{p.name}'}
@app.get('/api/exports/{filename}')
def download_export(filename:str):
 import re
 if not re.fullmatch(r'[0-9a-f]{32}-[0-9a-f]{10}-(?:all|coco|yolo|yolo-seg)\.zip',filename):raise HTTPException(404,'导出文件不存在')
 p=DATA/'exports'/filename
 if not p.is_file():raise HTTPException(404,'导出文件不存在')
 return FileResponse(p,media_type='application/zip',filename='PairLabel-'+filename[:8]+'.zip')
@app.get('/api/jobs/{jid}/export')
def export(jid:str,format:str='all',include_uncertain:bool=False,val_ratio:float=.2):
 p=build_export(jid,format,include_uncertain,val_ratio);return FileResponse(p,media_type='application/zip',filename=f'PairLabel-{jid[:8]}-{format}.zip')
@app.get('/api/jobs/{jid}/audit')
def get_audit(jid:str):require_job(jid);return db.rows('SELECT * FROM audit WHERE job_id=? ORDER BY id',(jid,))
app.mount('/',StaticFiles(directory=ROOT/'app/static',html=True),name='static')
