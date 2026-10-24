import copy,json,os
from pathlib import Path
from .profiles import is_xiangqi
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.getenv('LABELER_DATA',str(ROOT/'data'))).resolve()
CONFIG=ROOT/'config'/'config.json'
NAMES=[('king','帅','将'),('advisor','仕','士'),('elephant','相','象'),('horse','马','馬'),('rook','车','車'),('cannon','炮','砲'),('pawn','兵','卒')]
CLASSES=[]
for color,idx in [('red',1),('black',2)]:
 for kind,red,black in NAMES:
  CLASSES.append({'id':len(CLASSES),'name':f'{color}_{kind}','label':('红' if color=='red' else '黑')+(red if color=='red' else black),'color':color,'piece':kind})
CLASS_PRESETS={'general':[{'id':i,'name':name,'label':label} for i,(name,label) in enumerate([('person','人'),('cup','杯子'),('bottle','瓶子'),('mouse','鼠标')])],'xiangqi':copy.deepcopy(CLASSES)}
DEFAULT={
 'annotation_profile':'general','annotation_instructions':'','classes':copy.deepcopy(CLASS_PRESETS['general']),'iou_threshold':0.55,'min_confidence':0.80,'max_attempts':2,
 'retry_pixel_crops':True,'retry_tiles':True,'segmentation':True,'segmentation_backend':'sam2','sam2':{'weights':str(ROOT/'models/sam2.1_hiera_large.pt'),'threads':8,'min_score':0.85,'min_stability':0.90},'allow_empty':False,'circle_audit':True,
 'allowed_roots':[str(Path.home()/'workspace'),str(Path.home()/'Pictures'),str(Path.home()/'Downloads')],
 'runtime':{'binary':str(Path.home()/'ai/src/llama.cpp/build/bin/llama-server'),'port':18791,'context':16384,'image_tokens':4096,'timeout':1800,'idle_unload_seconds':180,
  'generator':{'name':'Qwen3.6-27B-Q6_K','family':'qwen3_5','weights':str(Path.home()/'ai/models/qwen3.6-27b/Qwen3.6-27B-Q6_K.gguf'),'projector':str(Path.home()/'ai/models/qwen3.6-27b/mmproj-F16.gguf')},
  'reviewer':{'name':'InternVL3.5-14B-Q8_0','family':'internvl','weights':str(ROOT/'models/InternVL3_5-14B.Q8_0.gguf'),'projector':str(ROOT/'models/InternVL3_5-14B.mmproj-f16.gguf')}}}

def validate(c):
 if not isinstance(c,dict):raise ValueError('配置必须是 JSON 对象')
 if c.get('annotation_profile','general') not in ('general','xiangqi'):raise ValueError('标注模式必须是 general 或 xiangqi')
 instructions=c.get('annotation_instructions','')
 if not isinstance(instructions,str) or len(instructions)>2000:raise ValueError('标注要求必须是字符串且不超过 2000 字')
 cs=c.get('classes',[])
 if not isinstance(cs,list) or not 1<=len(cs)<=200:raise ValueError('类别数量应为 1–200')
 names=set()
 for i,x in enumerate(cs):
  if not isinstance(x,dict) or type(x.get('id')) is not int or x['id']!=i or not isinstance(x.get('name'),str) or not x['name'] or x['name'] in names:raise ValueError('类别 ID 必须从 0 连续编号，英文名称不可重复')
  names.add(x['name'])
  if not isinstance(x.get('label'),str) or not x['label']:raise ValueError('每类需要中文显示名称')
  if any(v!=v.strip() or any(ch in v for ch in ('|','\n','\r')) for v in (x['name'],x['label'])):raise ValueError('类别名称及显示名称不可含竖线、换行或首尾空白；每行用 name | 显示名称填写')
 if not 0.1<=c.get('iou_threshold',0)<=0.95:raise ValueError('IoU 阈值应为 0.1–0.95')
 if not 0<=c.get('min_confidence',-1)<=1:raise ValueError('置信阈值应为 0–1')
 if not isinstance(c.get('max_attempts'),int) or not 1<=c['max_attempts']<=4:raise ValueError('尝试次数应为 1–4')
 rt=c['runtime']
 if rt['generator']['family']==rt['reviewer']['family']:raise ValueError('两个模型必须属于不同视觉架构')
 if Path(rt['generator']['weights']).resolve()==Path(rt['reviewer']['weights']).resolve():raise ValueError('生成和复核不能使用同一份权重')
 if not 1024<=int(rt['image_tokens'])<=12000:raise ValueError('视觉 token 数范围 1024–12000')
 return c

def load_config():
 c=copy.deepcopy(DEFAULT)
 if CONFIG.exists():
  user=json.loads(CONFIG.read_text());c.update(user)
  if 'annotation_profile' not in user:c['annotation_profile']='xiangqi' if is_xiangqi(user) else 'general'
 return validate(c)

def save_config(c):
 validate(c);CONFIG.parent.mkdir(exist_ok=True)
 tmp=CONFIG.with_suffix('.tmp');tmp.write_text(json.dumps(c,ensure_ascii=False,indent=2));tmp.replace(CONFIG)

def readiness(c=None):
 c=c or load_config();rt=c['runtime'];missing=[]
 for role in ['generator','reviewer']:
  for k in ['weights','projector']:
   p=Path(rt[role][k])
   if not p.is_file() or p.stat().st_size<1024:missing.append(f'{role}: {p.name}')
 if not Path(rt['binary']).is_file():missing.append('CUDA llama-server')
 if c.get('segmentation') and c.get('segmentation_backend','sam2')=='sam2':
  p=Path(c.get('sam2',{}).get('weights',str(ROOT/'models/sam2.1_hiera_large.pt')))
  if not p.is_file():missing.append('SAM 2.1 Large')
  import importlib.util
  if importlib.util.find_spec('sam2') is None:missing.append('SAM 2 Python 依赖')
 return {'ready':not missing,'missing':missing,'models':[{'role':r,**rt[r]} for r in ['generator','reviewer']]}
