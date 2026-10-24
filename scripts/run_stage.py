"""Operate one independent stage on a PAUSED job, preserving resumable evidence."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import db
from app.config import DATA
from app.inference import ModelManager
p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--role',choices=['generator','reviewer'],required=True);p.add_argument('--attempt',type=int,default=0);args=p.parse_args()
j=db.job(args.job)
if not j or j['status'] not in ['paused','failed']:raise RuntimeError('Job must be paused or failed before a standalone stage')
db.set_job(j['id'],'paused');c=j['config'];m=ModelManager()
try:
 for i in db.rows("SELECT * FROM items WHERE job_id=? AND status='pending' AND attempt=? ORDER BY name,id",(j['id'],args.attempt)):
  if db.one("SELECT id FROM runs WHERE item_id=? AND attempt=? AND role=? AND status='ok'",(i['id'],args.attempt,args.role)):continue
  db.audit(j['id'],i['id'],'inference_start',{'role':args.role,'attempt':args.attempt,'mode':'standalone_stage','independent':True})
  m.image_name=i['name']
  out=m.infer(args.role,DATA/i['path'],c,args.attempt);db.save_run(i['id'],args.attempt,args.role,c['runtime'][args.role]['name'],out);print(i['name'],len(out['prediction']['objects']),out['metrics']['seconds'],flush=True)
finally:m.stop()
