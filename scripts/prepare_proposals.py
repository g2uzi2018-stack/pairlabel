import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from app import db
from app.config import DATA
from app.profiles import is_xiangqi
from app.geometry import circle_candidates
from app.segmentation import Segmenter
p=db.item(sys.argv[1]);j=db.job(p['job_id'])
if not is_xiangqi(j['config']):raise ValueError('圆形候选工具仅用于象棋增强任务，通用模式请使用原图分块重试')
im=Image.open(DATA/p['path']).convert('RGB');seeds=[]
for x,y,r in circle_candidates(np.array(im)):
 seeds.append({'class_id':0,'bbox':[max(0,x-r*1.3),max(0,y-r*1.3),min(im.width,x+r*1.3),min(im.height,y+r*1.3)],'confidence':1,'polygon':[],'text':''})
objects,issues=Segmenter().segment(np.array(im),seeds,j['config']);good=[o for o in objects if o.get('segmentation_quality')=='heuristic_pass'];
for o in good:
 o['seed_bbox']=o.pop('vlm_bbox',o['bbox'])
 for key in ['class_id','confidence','text']:o.pop(key,None)
target=DATA/'jobs'/j['id']/'results'/f'{p["id"]}.proposals.json';target.write_text(db.dumps({'objects':good,'source':'class_blind_opencv_circles_plus_sam2_large','excluded':issues}));print(p['name'],len(seeds),'candidates',len(good),'reliable',issues,flush=True)
