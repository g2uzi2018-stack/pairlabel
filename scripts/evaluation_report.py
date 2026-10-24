"""Summarize actual saved predictions, keeping synthetic and public evidence separate."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import db
from app.config import DATA,ROOT,load_config
from scripts.benchmark import score
from app.geometry import iou
from scipy.optimize import linear_sum_assignment
import numpy as np
from PIL import Image,ImageDraw

def synthetic_mask_iou(result,truth,width,height):
 # six-pieces bodies are exactly the ellipses drawn in make_samples.py;
 # shadows are excluded. This is fixture truth, never used by inference.
 preds=result.get('objects',[]);gold=truth['objects']
 if not preds or not gold:return None
 costs=np.array([[1-iou(a['bbox'],b['bbox']) if a['class_id']==b['class_id'] else 2 for b in gold] for a in preds]);rr,cc=linear_sum_assignment(costs);values=[]
 for a,b in zip(rr,cc):
  if costs[a,b]>1 or not preds[a].get('polygon'):continue
  pm=Image.new('1',(width,height));ImageDraw.Draw(pm).polygon([tuple(p) for p in preds[a]['polygon']],fill=1)
  gm=Image.new('1',(width,height));ImageDraw.Draw(gm).ellipse(gold[b]['bbox'],fill=1)
  p=np.asarray(pm);g=np.asarray(gm);values.append(float(np.logical_and(p,g).sum()/np.logical_or(p,g).sum()))
 return {'matched_masks':len(values),'truth_masks':len(gold),'mean_iou':float(np.mean(values)) if values else None,'per_mask_iou':values,'truth_definition':'make_samples.py 棋子圆形本体，不含投影；仅生成图'}
report={'created':time.time(),'device':json.loads((DATA/'device.json').read_text()),'scope':'小规模功能验收，不是泛化准确率承诺','models':load_config()['runtime'],'baseline':[],'jobs':[],'warnings':['生成图和公开实拍图分别报告，不混成整体准确率。','公开样本部分只有棋子类型框，没有独立红黑类别真值。','SAM 掩膜自评分和稳定性不是人工分割真值。','数据量有限；不能证明对尚未拍摄的用户棋具达到同样效果。']}
for path in sorted((DATA/'benchmark/generator').glob('*.json')):
 d=json.loads(path.read_text());report['baseline'].append({'sample':path.stem,'synthetic':True,'model':'generator','score':d.get('score'),'inference_seconds':d.get('metrics',{}).get('seconds'),'wall_seconds':d.get('wall_seconds_including_load')})
for jid in json.loads((DATA/'validation-jobs.json').read_text()):
 j=db.job(jid);jr={'id':jid,'name':j['name'],'status':j['status'],'counts':j['counts'],'max_attempts':j['config']['max_attempts'],'items':[]}
 for row in db.rows('SELECT id FROM items WHERE job_id=?',(jid,)):
  it=db.item(row['id']);truthpath=ROOT/'samples'/('six-pieces.truth.json' if '生成' in it['name'] else 'public-six.truth.json');truth=json.loads(truthpath.read_text()) if truthpath.exists() and ('生成' in it['name'] or '堆叠' in it['name']) else None
  ir={'name':it['name'],'status':it['status'],'attempt':it['attempt'],'issues':it['issues'],'objects':len((it['result'] or {}).get('objects',[])),'runs':[]}
  if truth and it['result']:
   ir['final_score']=score(it['result'],truth)
   if '生成' in it['name']:ir['synthetic_mask_score']=synthetic_mask_iou(it['result'],truth,it['width'],it['height'])
  for run in db.rows('SELECT * FROM runs WHERE item_id=? ORDER BY attempt,role',(it['id'],)):
   p=json.loads(run['prediction']) if run['prediction'] else None;m=json.loads(run['metrics']);r={'role':run['role'],'attempt':run['attempt'],'status':run['status'],'model':run['model'],'seconds':m.get('seconds'),'predicted_objects':len(p.get('objects',[])) if p else None,'unknown_regions':len(p.get('unknown_regions',[])) if p else None,'error':run['error']}
   if truth and p:r['score']=score(p,truth)
   ir['runs'].append(r)
  jr['items'].append(ir)
 report['jobs'].append(jr)
checkpoint=DATA/'restart-checkpoint.json'
if checkpoint.exists():
 cp=json.loads(checkpoint.read_text());saved=db.one('SELECT * FROM runs WHERE id=?',(cp['row_id'],));audits=db.rows("SELECT * FROM audit WHERE item_id=? AND kind='inference_start' AND created>?",(cp['item_id'],cp['restart_time']));starts=[json.loads(a['detail']) for a in audits]
 report['actual_service_restart']={**cp,'saved_prediction_unchanged':bool(saved and saved['created']==cp['created'] and hashlib.sha256(saved['prediction'].encode()).hexdigest()==cp['prediction_sha256']),'generator_restarted_after_checkpoint':any(a['attempt']==cp['attempt'] and a['role']=='generator' for a in starts),'reviewer_continued_after_restart':any(a['attempt']==cp['attempt'] and a['role']=='reviewer' for a in starts)}
path=DATA/'evaluation-report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(path)
