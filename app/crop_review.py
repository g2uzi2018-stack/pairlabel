"""Class-blind pixel proposals; independent visual classification on contact sheets."""
import io,base64,json,copy,time,math,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import linear_sum_assignment
from . import db
from .config import DATA
from .taxonomy import normalize_taxonomy
from .profiles import is_xiangqi
from .prompts import build_crop_prompt
from .geometry import bounded_confidence,CONFIDENCE_SCHEMA
from .inference import http_json,InferenceError

def eligible(path,role,attempt,c):
 if not is_xiangqi(c):return None
 manifest=Path(path).parent.parent/'results'/(Path(path).stem+'.proposals.json')
 if not manifest.exists():return None
 props=json.loads(manifest.read_text());objects=props.get('objects',[])
 if not 1<=len(objects)<=60:return None
 previous=db.one("SELECT prediction FROM runs WHERE item_id=? AND role=? AND attempt=0 AND attempt<? AND status='ok' ORDER BY attempt ASC LIMIT 1",(Path(path).stem,role,attempt))
 if not previous:return None
 p,_=normalize_taxonomy(json.loads(previous['prediction']),c['classes'],enabled=is_xiangqi(c))
 # Every previous full-image object must be accounted for by nearby pixel evidence.
 prev=p['objects']
 if len(prev)>len(objects):return None
 if prev:
  costs=np.array([[np.hypot((o['bbox'][0]+o['bbox'][2]-q['bbox'][0]-q['bbox'][2])/2,(o['bbox'][1]+o['bbox'][3]-q['bbox'][1]-q['bbox'][3])/2)/max(8,o['bbox'][2]-o['bbox'][0],o['bbox'][3]-o['bbox'][1]) for q in objects] for o in prev])
  rr,cc=linear_sum_assignment(costs)
  if max(costs[rr,cc])>2:return None
 return props,p

def classify(manager,role,path,c,attempt,prepared):
 if not is_xiangqi(c):raise ValueError('候选拼图策略仅用于 xiangqi 模式')
 props,prior=prepared;im=Image.open(path).convert('RGB');objects=[];rejected=[];unknown=copy.deepcopy(prior.get('unknown_regions',[]));raws=[];metrics=[];started=time.time();cs=props['objects'];pages=(len(cs)+5)//6
 manager.ensure(role,c);rt=c['runtime'];manager.progress.update(tiles=pages,tile=1,strategy='pixel_crops')
 for page in range(pages):
  manager.progress['tile']=page+1;subset=cs[page*6:page*6+6];sheet=Image.new('RGB',(1536,1024),'#ecefea');draw=ImageDraw.Draw(sheet)
  for k,p in enumerate(subset):
   b=p['bbox'];bw=b[2]-b[0];bh=b[3]-b[1];pad=max(bw,bh)*.22
   bounds=(max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(im.width,int(b[2]+pad)),min(im.height,int(b[3]+pad)))
   crop=im.crop(bounds);scale=min(464/crop.width,440/crop.height);crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))),getattr(Image,'Resampling',Image).LANCZOS);x=(k%3)*512+(512-crop.width)//2;y=(k//3)*512+52+(440-crop.height)//2;sheet.paste(crop,(x,y));draw.text(((k%3)*512+16,(k//3)*512+14),f'PATCH {k+1}',fill='#17382a',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24))
  buf=io.BytesIO();sheet.save(buf,format='PNG')
  evidence=Path(path).parent.parent/'results'/f'{Path(path).stem}.crops.attempt-{attempt}.{role}.page-{page}.png';evidence.write_bytes(buf.getvalue());thumb=sheet.copy();thumb.thumbnail((600,400));thumb.save(evidence.with_suffix('.jpg'),quality=85)
  prompt=build_crop_prompt(c,len(subset))
  entry={'type':'object','properties':{'patch':{'type':'integer','minimum':1,'maximum':len(subset)},'kind':{'type':'string','enum':['piece','not_piece','unreadable']},'class_id':{'type':'integer','minimum':-1,'maximum':len(c['classes'])-1},'text':{'type':'string'},'color':{'type':'string','enum':['red','black','unknown']},'confidence':CONFIDENCE_SCHEMA,'reason':{'type':'string'}},'required':['patch','kind','class_id','text','color','confidence','reason'],'additionalProperties':False}
  schema={'type':'object','properties':{'items':{'type':'array','minItems':len(subset),'maxItems':len(subset),'items':entry}},'required':['items'],'additionalProperties':False}
  payload={'model':rt[role]['name'],'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}},{'type':'text','text':prompt}]}],'temperature':0,'seed':43+attempt,'max_tokens':2200,'response_format':{'type':'json_schema','json_schema':{'name':'patch_review','strict':True,'schema':schema}},'chat_template_kwargs':{'enable_thinking':False}}
  manager.state={'phase':'inference','message':f'{rt[role]["name"]} 独立读取候选棋字','role':role,'progress':manager.progress};t=time.time()
  result=http_json(f'http://127.0.0.1:{rt["port"]}/v1/chat/completions',payload,rt['timeout']);evidence.with_suffix('.json').write_text(json.dumps({'role':role,'model':rt[role],'attempt':attempt,'page':page,'prompt':prompt,'input_sha256':hashlib.sha256(buf.getvalue()).hexdigest(),'response':result},ensure_ascii=False,indent=2));choice=result['choices'][0];raw=choice['message'].get('content') or '';raws.append(raw)
  if choice.get('finish_reason')!='stop':raise InferenceError('候选区识别输出未完整结束',raw)
  try:items=json.loads(raw)['items']
  except (ValueError,KeyError) as e:raise InferenceError('候选区识别 JSON 无效',raw) from e
  if len(items)!=len(subset) or {x['patch'] for x in items}!=set(range(1,len(subset)+1)):raise InferenceError('候选区域编号不完整或重复',raw)
  for item in items:
   if item.get('color') not in ['red','black','unknown']:raise InferenceError('候选区颜色无效',raw)
   if type(item['patch']) is not int or type(item['class_id']) is not int or item['kind'] not in ['piece','not_piece','unreadable']:raise InferenceError('候选区类型无效',raw)
   conf_raw=item['confidence']
   try:item['confidence']=bounded_confidence(conf_raw)
   except ValueError as e:raise InferenceError('候选区置信值无效',raw) from e
   seed=subset[item['patch']-1];b=seed['bbox']
   if item['kind']=='piece' and item.get('color')=='unknown':
    unknown.append({'bbox':b,'reason':'候选棋子颜色无法判定'});continue
   if item['kind']=='piece':
    if not 0<=item['class_id']<len(c['classes']):raise InferenceError('棋子类别无效',raw)
    objects.append({'class_id':item['class_id'],'bbox':b,'confidence':float(item['confidence']),'text':item['text'],'color':item.get('color'),'polygon':seed.get('polygon',[]),'proposal_index':page*6+item['patch']-1,**({'confidence_raw':conf_raw,'numeric_normalization':'float_roundoff_to_unit_interval'} if conf_raw!=item['confidence'] else {})})
   elif item['kind']=='not_piece' and item['confidence']>=c['min_confidence']:
    rejected.append({'bbox':b,'confidence':item['confidence'],'text':item['text'],'reason':item['reason'],'proposal_index':page*6+item['patch']-1,**({'confidence_raw':conf_raw,'numeric_normalization':'float_roundoff_to_unit_interval'} if conf_raw!=item['confidence'] else {})})
   elif item['kind']=='unreadable' or item['confidence']<c['min_confidence']:
    unknown.append({'bbox':b,'reason':item['reason'] or '中心候选无法可靠归类'})
  metrics.append({'seconds':time.time()-t,'usage':result.get('usage'),'timings':result.get('timings'),'prompt':prompt,'page':page,'patches':subset,'input_sha256':hashlib.sha256(buf.getvalue()).hexdigest(),'evidence_file':str(evidence.relative_to(DATA))})
  manager.last_use=time.time()
 p,adjustments=normalize_taxonomy({'objects':objects,'unknown_regions':unknown,'image_quality':'独立候选区读字；共享像素定位','spatial_source':'sam2_pixel_proposals','rejected_proposals':rejected},c['classes'],enabled=is_xiangqi(c))
 if len(objects)<len(prior['objects']):p['unknown_regions'].append({'bbox':[0,0,im.width,im.height],'reason':'候选区检出数量低于本模型先前整图检出数量，需要核对漏标'})
 return {'prediction':p,'raw':json.dumps({'pages':raws},ensure_ascii=False),'metrics':{'seconds':time.time()-started,'independent_classification':True,'spatial_source':'shared_class_blind_pixel_proposals','retry_strategy':'sam2_crops','pages':metrics,'taxonomy_adjustments':adjustments}}
