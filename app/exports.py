import hashlib,json,zipfile,threading,shutil,contextlib
from pathlib import Path
from . import db
from .config import DATA
from .annotation_formats import yolo_text,coco_annotations,has_complete_polygons
from .job_access import job_guard

_export_locks={}
def export_busy(jid):return jid in _export_locks and _export_locks[jid].locked()
def build_export(jid,fmt='all',include_uncertain=False,val_ratio=.2):
 lock=_export_locks.setdefault(jid,threading.Lock())
 if not lock.acquire(blocking=False):raise ValueError('这项任务正在导出，请等待当前导出完成')
 try:
  with job_guard(jid):return _build_export(jid,fmt,include_uncertain,val_ratio)
 finally:lock.release()

@contextlib.contextmanager
def incomplete_archive(path):
 try:yield
 except Exception:
  path.unlink(missing_ok=True);raise

def _build_export(jid,fmt='all',include_uncertain=False,val_ratio=.2):
 job=db.job(jid)
 if not job:raise ValueError('任务不存在')
 if fmt not in ['all','yolo','yolo-seg','coco']:raise ValueError('不支持的格式')
 if not 0<=val_ratio<=.5:raise ValueError('验证集比例应为 0–0.5')
 if job['status'] in ['running','queued','pausing','importing']:raise ValueError('请先暂停任务，或等待任务完成后导出，以保证快照一致')
 classes=job['config']['classes'];items=db.rows('SELECT * FROM items WHERE job_id=? ORDER BY name,id',(jid,))
 selected=[i for i in items if i['status'] in (['accepted','approved','uncertain'] if include_uncertain else ['accepted','approved'])]
 if not selected:raise ValueError('没有符合条件的结果；默认只导出自动通过或人工确认的图片')
 destdir=DATA/'exports';destdir.mkdir(exist_ok=True)
 import uuid
 dest=destdir/f'{jid}-{uuid.uuid4().hex[:10]}-{fmt}.zip';temp=dest.with_suffix('.tmp')
 report={'job_id':jid,'name':job['name'],'classes':classes,'include_uncertain':include_uncertain,'val_ratio':val_ratio,'coordinate_system':'原图像素，EXIF 方向已统一；YOLO 导出归一化；COCO 类别 ID 从 1 开始','items':[],'segmentation_excluded':[],'quality_note':'双模型一致不是准确率证明。分割轮廓通过的是掩膜自评与启发式检查，未经人工真值校准。','config':job['config']}
 formats=['yolo','yolo-seg','coco'] if fmt=='all' else [fmt]
 estimated=sum((DATA/i['path']).stat().st_size for i in selected)*len(formats)
 if estimated>shutil.disk_usage(destdir).free*.9:raise ValueError('磁盘空间不足，请选择单一导出格式或缩小任务')
 with incomplete_archive(temp),zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
  coco={s:{'images':[],'annotations':[],'categories':[{'id':x['id']+1,'name':x['name'],'supercategory':x.get('color','object')} for x in classes]} for s in ['train','val']};ann_id=1
  for f in formats:
   if f.startswith('yolo'):
    for s in ['train','val']:
     z.writestr(f'{f}/images/{s}/','');z.writestr(f'{f}/labels/{s}/','')
    # Omit path: Ultralytics then resolves images against this YAML's directory.
    z.writestr(f'{f}/dataset.yaml','train: images/train\nval: images/val\nnames:\n'+'\n'.join(f'  {x["id"]}: '+json.dumps(x['name'],ensure_ascii=False) for x in classes)+'\n')
  for image_id,i in enumerate(selected,1):
   result=json.loads(i['result'] or '{"objects":[]}');objs=result['objects'];w=i['width'];h=i['height'];split='val' if int(i['hash'][:8],16)/0xffffffff<val_ratio else 'train';filename=i['id']+'.png'
   record={k:i[k] for k in ['id','name','hash','width','height','status']};record.update(split=split,objects=len(objs));report['items'].append(record)
   for f in formats:
    if f=='yolo-seg' and not has_complete_polygons(objs):
     report['segmentation_excluded'].append({'id':i['id'],'reason':'缺少可靠轮廓，跳过整张图，避免丢目标'});continue
    z.write(DATA/i['path'],f'{f}/images/{split}/{filename}')
    if f=='yolo':
     z.writestr(f'{f}/labels/{split}/{i["id"]}.txt',yolo_text(objs,w,h))
    elif f=='yolo-seg':
     z.writestr(f'{f}/labels/{split}/{i["id"]}.txt',yolo_text(objs,w,h,segmentation=True))
   if 'coco' in formats:
    coco[split]['images'].append({'id':image_id,'file_name':f'images/{split}/{filename}','width':w,'height':h,'original_name':i['name']})
    coco[split]['annotations'].extend(coco_annotations(objs,image_id,ann_id));ann_id+=len(objs)
   preview=DATA/'jobs'/jid/'results'/f'{i["id"]}.jpg'
   if preview.exists():z.write(preview,f'previews/{i["id"]}.jpg')
  if 'coco' in formats:
   for s,d in coco.items():z.writestr(f'coco/annotations/instances_{s}.json',json.dumps(d,ensure_ascii=False))
  # Preserve all attempts and human actions, even for items excluded from training.
  for p in sorted((DATA/'jobs'/jid/'results').glob('*.crops.*.png')):z.write(p,f'review/crops/{p.name}')
  for p in sorted((DATA/'jobs'/jid/'results').glob('*.json')):z.write(p,f'review/{p.name}')
  audit=db.rows('SELECT * FROM audit WHERE job_id=? ORDER BY id',(jid,));z.writestr('review/audit.jsonl','\n'.join(db.dumps(x) for x in audit)+'\n')
  z.writestr('manifest.json',json.dumps(report,ensure_ascii=False,indent=2))
  z.writestr('README.txt','象棋图片本地自动标注导出\n默认只含自动通过 / 人工确认结果。原始文件名见 manifest.json。\nYOLO 检测与分割位于独立目录，类别 ID 从 0 起。COCO 类别 ID 从 1 起。\n训练/验证按图片内容哈希固定划分，同图不会跨集合。小数据集可能某一集合为空。\nreview/ 保存双模型独立输出、分歧和人工操作记录。\n分割缺失的图片整体排除于 YOLO-seg，详见 manifest.json。\n模型一致不代表已证明正确；正式训练前应抽样核对真实图片。\n')
 temp.replace(dest);db.audit(jid,None,'export',{'format':fmt,'include_uncertain':include_uncertain,'images':len(selected),'file':dest.name});return dest
