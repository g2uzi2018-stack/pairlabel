"""SAM 2.1 Large with explicit CPU/CUDA selection and FP32 quality checks."""
import copy,gc,time
from pathlib import Path
import cv2
import numpy as np
from .config import ROOT
from .geometry import iou,segment as grabcut_segment

def select_device(requested,torch):
 if requested not in ['auto','cpu','cuda']:raise ValueError('SAM device 必须是 auto、cpu 或 cuda')
 if requested=='cpu':return 'cpu'
 cuda=getattr(torch,'cuda',None)
 available=bool(cuda is not None and cuda.is_available())
 if requested=='cuda' and not available:raise ValueError('SAM 请求 CUDA，但当前 PyTorch 或设备不支持；请使用 GPU 环境，或显式选择 cpu')
 return 'cuda' if available else 'cpu'

class Segmenter:
 def __init__(self):self.predictor=None;self.signature=None;self.image_ref=None;self.device=None
 def unload(self):
  device=self.device
  self.predictor=None;self.signature=None;self.image_ref=None;self.device=None;gc.collect()
  if device=='cuda':
   import torch
   torch.cuda.empty_cache()
 def segment(self,rgb,objects,c):
  backend=c.get('segmentation_backend','sam2')
  if backend=='grabcut':return grabcut_segment(rgb,objects)
  if not objects:return [],[]
  if backend!='sam2':raise ValueError('分割后端必须是 sam2 或 grabcut')
  cfg=c.get('sam2',{});weights=Path(cfg.get('weights',str(ROOT/'models/sam2.1_hiera_large.pt')))
  if not weights.exists():raise ValueError('SAM 2.1 Large 权重未就绪；请运行 scripts/download_sam.py')
  import torch
  from sam2.build_sam import build_sam2
  from sam2.sam2_image_predictor import SAM2ImagePredictor
  device=select_device(cfg.get('device','auto'),torch)
  signature=str(weights.resolve())+(':cuda_fp32' if device=='cuda' else '')
  if self.predictor is None or self.signature!=signature:
   self.unload()
   torch.set_num_threads(int(cfg.get('threads',8)))
   if device=='cuda':
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
   self.predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(weights),device=device,apply_postprocessing=False));self.signature=signature;self.image_ref=None;self.device=device
  started=time.time();out=copy.deepcopy(objects);issues=[]
  with torch.inference_mode():
   if self.image_ref is not rgb:
    self.predictor.set_image(np.array(rgb,copy=True));self.image_ref=rgb
   for idx,o in enumerate(out):
    b=o['bbox']
    try:
     logits,scores,_=self.predictor.predict(box=np.array(b,dtype=np.float32),multimask_output=True,return_logits=True)
     best=int(np.argmax(scores));values=logits[best];mask=(values>0).astype('uint8');score=float(scores[best]);stability=float((values>1).sum()/max(1,(values>-1).sum()))
     contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     if not contours:raise ValueError('没有前景')
     cnt=max(contours,key=cv2.contourArea);area=cv2.contourArea(cnt);total=mask.sum();x,y,w,h=cv2.boundingRect(cnt);mask_box=[x,y,x+w,y+h]
     approx=cv2.approxPolyDP(cnt,max(.75,.0015*cv2.arcLength(cnt,True)),True).reshape(-1,2)
     reason=[]
     if score<float(cfg.get('min_score',.85)):reason.append(f'掩膜自评 {score:.2f}')
     if stability<float(cfg.get('min_stability',.90)):reason.append(f'阈值稳定性 {stability:.2f}')
     if iou(mask_box,b)<.55:reason.append('轮廓外接框与视觉模型定位差异较大')
     if area<.25*(b[2]-b[0])*(b[3]-b[1]):reason.append('掩膜面积过小')
     if area/max(1,total)<.8:reason.append('存在较大分离前景')
     if len(approx)<4:reason.append('轮廓点不足')
     o.update(vlm_bbox=list(b),bbox=mask_box,polygon=[[int(x),int(y)] for x,y in approx],segmentation_method='sam2.1_hiera_large_'+('cuda_fp32' if device=='cuda' else 'cpu'),mask_area=round(area,2),segmentation_score=round(score,4),segmentation_stability=round(stability,4),segmentation_quality='uncertain' if reason else 'heuristic_pass')
     if reason:issues.append({'type':'segmentation','message':f'目标 {idx+1} 分割需确认：'+ '；'.join(reason),'generator':idx})
    except (ValueError,RuntimeError) as e:
     o.update(vlm_bbox=list(b),bbox=list(b),polygon=[],segmentation_method='sam2.1_hiera_large_'+('cuda_fp32' if device=='cuda' else 'cpu'),segmentation_quality='uncertain');issues.append({'type':'segmentation','message':f'目标 {idx+1} 分割失败：{str(e)[:150]}','generator':idx})
  duration=round(time.time()-started,3)
  for o in out:o['segmentation_image_seconds']=duration
  return out,issues
