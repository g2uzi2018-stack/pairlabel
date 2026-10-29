import json,threading,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import db
from .config import DATA,ROOT,readiness
from .inference import ModelManager,InferenceError
from .geometry import compare,circle_candidates,audit_circles,iou
from .segmentation import Segmenter
from .taxonomy import normalize_taxonomy
from .profiles import is_xiangqi
from .classification import unreadable

COLORS=['#e45b4b','#3988c4','#20a47d','#ae7bdb','#d58c29','#21a2b5','#d85d97']
def preview(path,result,classes,target):
 im=Image.open(path).convert('RGB');draw=ImageDraw.Draw(im,'RGBA');font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
 try:font=ImageFont.truetype(font_path,max(14,min(im.size)//45))
 except OSError:font=ImageFont.load_default()
 for idx,o in enumerate(result['objects']):
  camp=classes[o['class_id']].get('color');color='#888888' if unreadable(o) else '#df5a4e' if camp=='red' else '#456277' if camp=='black' else COLORS[o['class_id']%len(COLORS)];b=o['bbox']
  if o.get('polygon'):draw.polygon([tuple(p) for p in o['polygon']],fill=color+'35',outline=color)
  draw.rectangle(b,outline=color,width=max(2,min(im.size)//350))
  label=f'{idx+1} '+('看不清' if unreadable(o) else classes[o['class_id']]['label'])
  try:draw.text((b[0]+2,max(0,b[1]-22)),label,font=font,fill=color,stroke_width=1,stroke_fill='white')
  except UnicodeEncodeError:draw.text((b[0]+2,max(0,b[1]-15)),'?' if unreadable(o) else str(o['class_id']),font=font,fill=color)
 im.thumbnail((2000,2000));im.save(target,format='JPEG',quality=92)

class Worker:
 def __init__(self):self.segmenter=Segmenter();self.manager=ModelManager();self.wake=threading.Event();self.stopping=threading.Event();self.thread=None
 def start(self):
  self.thread=threading.Thread(target=self.loop,name='label-worker',daemon=True);self.thread.start()
 def stop(self):
  self.stopping.set();self.wake.set()
  # Cancellation terminates only this worker's model subprocess.
  self.manager.stop()
  if self.thread:self.thread.join(timeout=10)
 def is_active(self,jid):return not self.stopping.is_set() and bool(db.one("SELECT id FROM jobs WHERE id=? AND status='running' AND deleted_at=0",(jid,)))
 def loop(self):
  while not self.stopping.is_set():
   j=db.one("SELECT id FROM jobs WHERE status='queued' AND deleted_at=0 ORDER BY created LIMIT 1")
   if not j:
    if self.manager.proc and time.time()-self.manager.last_use>180:self.manager.stop()
    self.wake.wait(2);self.wake.clear();continue
   jid=j['id'];job=db.job(jid)
   if not job:continue
   config=job['config']
   if not readiness(config)['ready']:
    db.set_job(jid,'failed','模型文件未就绪；下载完成后点击继续');continue
   with db.connection() as con:
    claimed=con.execute("UPDATE jobs SET status='running',updated=? WHERE id=? AND status='queued' AND deleted_at=0",(time.time(),jid)).rowcount
   if not claimed:continue
   try:self.process(jid,config)
   except Exception as e:
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/worker.log').open('a') as f:f.write(traceback.format_exc()+'\n')
    current_status=db.one('SELECT status FROM jobs WHERE id=?',(jid,))['status']
    next_status=('paused' if current_status in ['paused','pausing'] else 'queued') if self.stopping.is_set() else 'failed'
    db.set_job(jid,next_status,str(e)[:1800]);self.manager.stop()
   finally:
    if db.one('SELECT status FROM jobs WHERE id=?',(jid,))['status']=='pausing':db.set_job(jid,'paused')
    if not db.one("SELECT id FROM jobs WHERE status='queued'"):
     self.manager.stop();self.segmenter.unload()
 def process(self,jid,c):
  for attempt in range(c['max_attempts']):
   if not self.is_active(jid):return
   # Each model processes a batch independently. No peer output enters either request.
   candidates=db.rows("SELECT id,path,name FROM items WHERE job_id=? AND status='pending' AND attempt=? ORDER BY name,id",(jid,attempt))
   if not candidates:continue
   for role in ['generator','reviewer']:
    for row in candidates:
     if not self.is_active(jid):return
     old=db.one('SELECT status FROM runs WHERE item_id=? AND attempt=? AND role=?',(row['id'],attempt,role))
     if old and old['status']=='ok':continue
     db.audit(jid,row['id'],'inference_start',{'role':role,'attempt':attempt,'model':c['runtime'][role]['name']})
     try:
      self.manager.image_name=row['name']
      out=self.manager.infer(role,DATA/row['path'],c,attempt)
      db.save_run(row['id'],attempt,role,c['runtime'][role]['name'],out)
     except InferenceError as e:
      db.save_run(row['id'],attempt,role,c['runtime'][role]['name'],{'raw':e.raw,'metrics':e.metrics},str(e))
      db.audit(jid,row['id'],'inference_error',{'role':role,'attempt':attempt,'error':str(e)})
      # A process that failed to load cannot produce any result; stop the batch.
      if not self.manager.loaded:raise
   for row in candidates:
    if not self.is_active(jid):return
    self.finalize(jid,row['id'],attempt,c)
  if self.is_active(jid):db.set_job(jid,'completed')
 def finalize(self,jid,iid,attempt,c):
  item=db.item(iid);runs=db.rows('SELECT * FROM runs WHERE item_id=? AND attempt=? ORDER BY role',(iid,attempt));by={r['role']:r for r in runs}
  chess=is_xiangqi(c)
  intrinsic=False;issues=[];matches=[];result={'objects':[],'matches':[],'attempt':attempt,'independence':'两个模型在互不可见结果的条件下识别原图'}
  for role in ['generator','reviewer']:
   r=by.get(role)
   if not r or r['status']!='ok':issues.append({'type':'inference_error','message':f'{role}: '+(r['error'] if r else '推理未完成')})
  if not issues:
   a,adj_a=normalize_taxonomy(json.loads(by['generator']['prediction']),c['classes'],enabled=chess);b,adj_b=normalize_taxonomy(json.loads(by['reviewer']['prediction']),c['classes'],enabled=chess);result['taxonomy_adjustments']={'generator':adj_a,'reviewer':adj_b}
   # General classes may carry arbitrary metadata; chess-only diagnostics must
   # not affect comparison. Original model evidence remains in a, b and runs.
   compared=[{**p,'objects':[{k:v for k,v in o.items() if chess or k not in ('color','taxonomy_ambiguous')} for o in p['objects']]} for p in (a,b)]
   issues,matches=compare(*compared,c)
   if chess:
    backs=lambda p:[r for r in p.get('unknown_regions',[]) if any(k in r.get('reason','') for k in ['背面','无字','face-down','back side'])]
    intrinsic=any(iou(x['bbox'],y['bbox'])>.35 for x in backs(a) for y in backs(b))
    if intrinsic:result['retry_skipped']='两个模型均报告同一背面或无字目标；原图缺少可读文字，直接保留为疑难，避免无效重复推理。'
    elif any(not any(iou(r['bbox'],o['bbox'])>.2 for o in other['objects']) for source,other in [(a,b),(b,a)] for r in backs(source)):
     intrinsic=True;result['retry_skipped']='一方报告背面或无字，另一方未在该处识别出可读类别；按保守规则直接保留疑难，仍可手动重试。'
   result['objects']=a['objects'];result['matches']=matches
   rgb=np.asarray(Image.open(DATA/item['path']).convert('RGB'))
   if chess and c.get('circle_audit'):
    circles=circle_candidates(rgb);result['circle_candidates']=circles
    def rejected_by(p,x,y):return any(q['bbox'][0]<=x<=q['bbox'][2] and q['bbox'][1]<=y<=q['bbox'][3] for q in p.get('rejected_proposals',[]))
    ignored=[q for q in circles if rejected_by(a,q[0],q[1]) and rejected_by(b,q[0],q[1])];result['independently_rejected_circles']=ignored
   # Chess inventory applies only to the shipped Chinese-chess taxonomy.
   if chess:
    limits={'king':1,'advisor':2,'elephant':2,'horse':2,'rook':2,'cannon':2,'pawn':5};counts={}
    for o in result['objects']:counts[o['class_id']]=counts.get(o['class_id'],0)+1
    for cid,n in counts.items():
     if n>limits.get(c['classes'][cid].get('piece'),999):issues.append({'type':'inventory','message':f'{c["classes"][cid]["label"]} 数量 {n} 超过标准棋局上限；可能误读，也可能是摆拍'})
   if c.get('segmentation'):
    self.manager.state={'phase':'segmentation','message':'SAM 2.1 Large 正在检查轮廓','role':None,'progress':{'image':item['name'],'attempt':attempt+1,'tile':1,'tiles':1,'started':time.time()}}
    try:
     result['objects'],segissues=self.segmenter.segment(rgb,result['objects'],c);issues+=segissues
     if c.get('retry_pixel_crops',True) and c.get('segmentation_backend','sam2')=='sam2' and chess:
      circles=circle_candidates(rgb);seeds=[]
      for x,y,r in circles:
       bb=[max(0,x-r*1.3),max(0,y-r*1.3),min(item['width'],x+r*1.3),min(item['height'],y+r*1.3)]
       seeds.append({'class_id':0,'bbox':bb,'confidence':1,'polygon':[],'text':''})
      grounded,ground_issues=self.segmenter.segment(rgb,seeds,c) if seeds else ([],[])
      peer_grounded,peer_issues=self.segmenter.segment(rgb,b['objects'],c)
      pool=[o for o in [*result['objects'],*peer_grounded,*grounded] if o.get('segmentation_quality')=='heuristic_pass'];proposals=[]
      for o in sorted(pool,key=lambda q:(q.get('segmentation_score',0),q.get('mask_area',0)),reverse=True):
       if any(iou(o['bbox'],q['bbox'])>.65 for q in proposals):continue
       # Strip all model categories: the two crop readers receive pixels only.
       o={**o};o['seed_bbox']=o.pop('vlm_bbox',o['bbox'])
       for key in ['class_id','confidence','text','color','class_id_raw']:o.pop(key,None)
       proposals.append(o)
      pp=DATA/'jobs'/jid/'results'/f'{iid}.proposals.json'
      pp.write_text(db.dumps({'objects':proposals,'source':'class_blind_sam2_union_of_independent_detections_and_circles','excluded':[*ground_issues,*peer_issues]}))
     if a.get('spatial_source') or b.get('spatial_source'):
      result['spatial_note']='候选框来自共享像素分割，两个模型独立判断类别；相同框的 IoU 不作为独立定位证据，定位依据为 SAM 掩膜质量检查。'
    except (ValueError,RuntimeError,ImportError) as e:
     issues.append({'type':'segmentation_unavailable','message':str(e)[:300]})
    # Mask refinement can snap distinct VLM seeds onto the same physical piece.
    # Recheck final geometry rather than trusting only pre-segmentation boxes.
    for n,o in enumerate(result['objects']):
     for k,other in enumerate(result['objects'][:n]):
      if iou(o['bbox'],other['bbox'])>.65:issues.append({'type':'mask_overlap','message':f'分割后目标 {n+1} 与 {k+1} 高度重叠，可能定位到同一个实体'})
   if chess and c.get('circle_audit'):issues+=audit_circles(result['objects'],[q for q in result['circle_candidates'] if q not in result['independently_rejected_circles']])
  elif 'generator' in by and by['generator']['status']=='ok':result['objects']=json.loads(by['generator']['prediction'])['objects']
  result['issues']=issues
  retry=bool(issues) and attempt+1<c['max_attempts'] and not intrinsic
  status='pending' if retry else ('uncertain' if issues else 'accepted')
  next_attempt=attempt+1 if retry else attempt
  itemdir=DATA/'jobs'/jid/'results';itemdir.mkdir(exist_ok=True)
  # Preserve each attempt, including failed and superseded evidence.
  report={'item_id':iid,'name':item['name'],'image_hash':item['hash'],'status':status,'result':result,'runs':[{**r,'prediction':json.loads(r['prediction']) if r['prediction'] else None,'metrics':json.loads(r['metrics'])} for r in runs]}
  target=itemdir/f'{iid}.attempt-{attempt}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(target)
  preview(DATA/item['path'],result,c['classes'],itemdir/f'{iid}.jpg')
  db.execute('UPDATE items SET status=?,attempt=?,result=?,issues=?,error=? WHERE id=?',(status,next_attempt,db.dumps(result),db.dumps(issues),'',iid))
  db.audit(jid,iid,'comparison',{'attempt':attempt,'status':status,'issues':issues,'matches':matches})
  # JSONL is an inspectable mirror; SQLite is the source of truth.
  with (DATA/'jobs'/jid/'review.jsonl').open('a') as f:f.write(db.dumps(report)+'\n')
