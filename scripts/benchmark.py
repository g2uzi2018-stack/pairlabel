#!/usr/bin/env python3
"""Runs genuine local vision inference against explicit synthetic truth."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import ROOT,DATA,CLASSES,load_config
from app.inference import ModelManager
from app.geometry import iou
import numpy as np
from scipy.optimize import linear_sum_assignment

def score(pred,truth):
 a=pred['objects'];b=truth['objects'];tp=0;matched=[]
 if a and b:
  scores=np.array([[iou(x['bbox'],y['bbox']) if x['class_id']==y['class_id'] else 0 for y in b] for x in a]);rr,cc=linear_sum_assignment(1-scores)
  for r,c in zip(rr,cc):
   if scores[r,c]>=.5:tp+=1;matched.append(float(scores[r,c]))
 return {'true_positives_iou50':tp,'predicted':len(a),'truth':len(b),'precision_iou50':tp/len(a) if a else (1 if not b else 0),'recall_iou50':tp/len(b) if b else (1 if not a else 0),'mean_matched_iou':sum(matched)/len(matched) if matched else None}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--role',choices=['generator','reviewer'],required=True);p.add_argument('--samples',nargs='+',default=['six-pieces','rotated-pieces','full-board','empty']);args=p.parse_args()
 dest=DATA/'benchmark'/args.role;dest.mkdir(parents=True,exist_ok=True);c=load_config();c.update(annotation_profile='xiangqi',annotation_instructions='',classes=copy.deepcopy(CLASSES));manager=ModelManager()
 try:
  for name in args.samples:
   t=time.time();result=manager.infer(args.role,ROOT/'samples'/f'{name}.png',c,0);result['wall_seconds_including_load']=round(time.time()-t,2);result['score']=score(result['prediction'],json.loads((ROOT/'samples'/f'{name}.truth.json').read_text()));result['synthetic']=True
   (dest/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(name,result['score'],result['wall_seconds_including_load'],flush=True)
 finally:manager.stop()
