import fcntl,base64,hashlib,io,json,os,signal,subprocess,threading,time,urllib.error,urllib.request
from pathlib import Path
from PIL import Image,ImageEnhance
from .config import ROOT,DATA
from .geometry import validate_prediction,iou,CONFIDENCE_SCHEMA
from .taxonomy import normalize_taxonomy
from .profiles import is_xiangqi
from .prompts import build_detection_prompt

class InferenceError(RuntimeError):
 def __init__(self,message,raw='',metrics=None):super().__init__(message);self.raw=raw;self.metrics=metrics or {}

def http_json(url,payload=None,timeout=5):
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 data=json.dumps(payload).encode() if payload is not None else None
 req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
 with op.open(req,timeout=timeout) as r:return json.load(r)

class ModelManager:
 def __init__(self):
  self.progress={};self.image_name=None;self.file_lock=None;self.proc=None;self.loaded=None;self.lock=threading.RLock();self.state={'phase':'idle','message':'等待任务','role':None};self.last_use=time.time();self.log=None
 def stop(self):
  with self.lock:
   if self.proc:
    self.proc.terminate()
    try:self.proc.wait(timeout=20)
    except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=10)
    self.proc=None
   if self.log:self.log.close();self.log=None
   if self.file_lock:
    fcntl.flock(self.file_lock,fcntl.LOCK_UN);self.file_lock.close();self.file_lock=None
   self.loaded=None;self.state={'phase':'idle','message':'模型已卸载','role':None}
 def ensure(self,role,c):
  rt=c['runtime'];model=rt[role];sig=(role,json.dumps(rt,sort_keys=True))
  with self.lock:
   if self.proc and self.proc.poll() is None and self.loaded==sig:return
   self.stop()
   for k in ['weights','projector']:
    if not Path(model[k]).is_file():raise InferenceError(f'模型文件未就绪：{Path(model[k]).name}')
   DATA.mkdir(parents=True,exist_ok=True)
   self.file_lock=(DATA/'inference.lock').open('a')
   try:fcntl.flock(self.file_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:
    self.file_lock.close();self.file_lock=None;raise InferenceError('另一个本应用推理进程正在运行；结束基准测试后再继续任务')
   self.state={'phase':'loading','message':f'加载 {model["name"]}，可能需要数分钟','role':role,'progress':self.progress}
   logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
   self.log=(logdir/f'{role}-server.log').open('a')
   cmd=[rt['binary'],'--model',model['weights'],'--mmproj',model['projector'],'--host','127.0.0.1','--port',str(rt['port']),'--alias',model['name'],'--jinja','--reasoning','off','--no-ui','--gpu-layers','all','--ctx-size',str(rt['context']),'--flash-attn','on','--parallel','1','--image-min-tokens','1024','--image-max-tokens',str(rt['image_tokens']),'--no-warmup']
   env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()};env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
   # Fail on a conflicting port before starting, never adopt unrelated services.
   import socket
   with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',rt['port']))==0:raise InferenceError(f'推理端口 {rt["port"]} 被其他服务占用')
   self.proc=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   deadline=time.time()+600
   while time.time()<deadline:
    if self.proc.poll() is not None:
     code=self.proc.returncode;self.stop();raise InferenceError(f'{model["name"]} 加载失败（{code}），请查看 logs/{role}-server.log')
    try:
     if http_json(f'http://127.0.0.1:{rt["port"]}/health').get('status')=='ok':
      self.loaded=sig;self.last_use=time.time();self.state={'phase':'ready','message':f'{model["name"]} 已就绪','role':role,'progress':self.progress};return
    except (OSError,ValueError):pass
    time.sleep(1)
   self.stop();raise InferenceError('模型加载超过 10 分钟')
 def infer(self,role,path,c,attempt):
  self.progress={'image':self.image_name or Path(path).name,'attempt':attempt+1,'tile':1,'tiles':4 if attempt and c.get('retry_tiles',True) else 1,'started':time.time()}
  if attempt and c.get('retry_pixel_crops',True):
   from .crop_review import eligible,classify
   prepared=eligible(path,role,attempt,c)
   if prepared:
    try:return classify(self,role,path,c,attempt,prepared)
    except InferenceError:raise
    except (OSError,ValueError,KeyError,IndexError) as e:
     if isinstance(e,(TimeoutError,urllib.error.URLError)):self.stop()
     raise InferenceError('候选区识别失败：'+str(e)) from e
  if not attempt or not c.get('retry_tiles',True):return self._infer_single(role,path,c,attempt)
  original=Image.open(path).convert('RGB');w,h=original.size;objects=[];unknown=[];tiles=[];started=time.time()
  try:
   for row in range(2):
    for col in range(2):
     self.progress['tile']=row*2+col+1
     core=[col*w/2,row*h/2,(col+1)*w/2,(row+1)*h/2]
     bounds=[max(0,int(core[0]-.18*w)),max(0,int(core[1]-.18*h)),min(w,int(core[2]+.18*w)),min(h,int(core[3]+.18*h))]
     out=self._infer_single(role,path,c,attempt,original.crop(bounds));tiles.append({'bounds':bounds,**out})
     for region in out['prediction'].get('unknown_regions',[]):
      b=region['bbox'];unknown.append({'bbox':[b[0]+bounds[0],b[1]+bounds[1],b[2]+bounds[0],b[3]+bounds[1]],'reason':region['reason']})
     for o in out['prediction']['objects']:
      o=dict(o);b=o['bbox'];b=[b[0]+bounds[0],b[1]+bounds[1],b[2]+bounds[0],b[3]+bounds[1]];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
      if not(core[0]<=cx<core[2] and core[1]<=cy<core[3]):continue
      o['bbox']=b;o['source_tile']=bounds
      if any(o['class_id']==prev['class_id'] and iou(b,prev['bbox'])>.65 for prev in objects):continue
      objects.append(o)
   return {'prediction':{'objects':objects,'unknown_regions':unknown,'image_quality':'overlapping_2x2_independent_retry'},'raw':json.dumps({'tiles':[{'bounds':t['bounds'],'raw':t['raw']} for t in tiles]},ensure_ascii=False),'metrics':{'seconds':round(time.time()-started,3),'independent':True,'retry_strategy':'overlapping_2x2_original_image','tiles':[{'bounds':t['bounds'],**t['metrics']} for t in tiles]}}
  except InferenceError as e:
   raise InferenceError('分块重试未全部完成：'+str(e),json.dumps({'completed_tiles':tiles,'failed_tile_raw':e.raw},ensure_ascii=False),{'seconds':time.time()-started}) from e
 def _infer_single(self,role,path,c,attempt,image=None):
  self.ensure(role,c);self.last_use=time.time();rt=c['runtime'];model=rt[role]
  image=image if image is not None else Image.open(path).convert('RGB');w,h=image.size
  # Retry modifies only the shared original image, never exposes the peer output.
  if attempt>0:image=ImageEnhance.Contrast(image).enhance(1.12)
  image.thumbnail((2200,2200),getattr(Image,'Resampling',Image).LANCZOS)
  out=io.BytesIO();image.save(out,format='PNG')
  prompt=build_detection_prompt(c,attempt)
  schema={'type':'object','properties':{'objects':{'type':'array','maxItems':300,'items':{'type':'object','properties':{'class_id':{'type':'integer','minimum':0,'maximum':len(c['classes'])-1},'bbox':{'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':4,'maxItems':4},'confidence':CONFIDENCE_SCHEMA,'text':{'type':'string'}},'required':['class_id','bbox','confidence','text'],'additionalProperties':False}},'unknown_regions':{'type':'array','items':{'type':'object','properties':{'bbox':{'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':4,'maxItems':4},'reason':{'type':'string'}},'required':['bbox','reason'],'additionalProperties':False}},'image_quality':{'type':'string'}},'required':['objects','unknown_regions','image_quality'],'additionalProperties':False}
  if is_xiangqi(c):
   object_schema=schema['properties']['objects']['items'];object_schema['properties']['color']={'type':'string','enum':['red','black','unknown']};object_schema['required'].append('color')
  request={'model':model['name'],'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()}},{'type':'text','text':prompt}]}],'temperature':0,'seed':42+attempt,'max_tokens':6500,'response_format':{'type':'json_schema','json_schema':{'name':'annotation','strict':True,'schema':schema}},'chat_template_kwargs':{'enable_thinking':False}}
  self.state={'phase':'inference','message':f'{model["name"]} 正在独立识别','role':role,'progress':self.progress}
  started=time.time();raw='';metrics={}
  try:
   result=http_json(f'http://127.0.0.1:{rt["port"]}/v1/chat/completions',request,timeout=rt['timeout'])
   choice=result['choices'][0];raw=choice['message'].get('content') or ''
   metrics={'seconds':round(time.time()-started,3),'usage':result.get('usage',{}),'timings':result.get('timings',{}),'finish_reason':choice.get('finish_reason'),'model':model,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'attempt':attempt,'image_size':[w,h],'sent_size':list(image.size),'independent':True}
   if choice.get('finish_reason')!='stop':raise InferenceError('输出截断或非正常结束，不能作为完整标注',raw,metrics)
   text=raw.strip()
   if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
   prediction=validate_prediction(json.loads(text),c['classes'],w,h)
   prediction,adjustments=normalize_taxonomy(prediction,c['classes'],enabled=is_xiangqi(c));metrics['taxonomy_adjustments']=adjustments
   return {'prediction':prediction,'raw':raw,'metrics':metrics}
  except InferenceError:raise
  except (OSError,ValueError,KeyError,IndexError) as e:
   # A timed out server may still be generating. Unload to cancel before retry.
   if isinstance(e,(TimeoutError,urllib.error.URLError)):self.stop()
   raise InferenceError(str(e)[:1500],raw,metrics) from e
  finally:self.last_use=time.time()
