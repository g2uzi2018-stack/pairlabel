import copy,math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# llama.cpp's number min/max are not grammar bounds. Enumerate unit-interval
# self-ratings so constrained decoding cannot emit a 1–5 or percentage score.
CONFIDENCE_SCHEMA={'type':'number','enum':[i/100 for i in range(101)]}

def iou(a,b):
 w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=w*h
 return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def bounded_confidence(value):
 if type(value) not in (int,float) or not math.isfinite(value) or not -1e-6<=value<=1+1e-6:raise ValueError('无效置信值')
 return max(0.0,min(1.0,float(value)))

def validate_prediction(p,classes,w,h,normalized=True):
 if not isinstance(p,dict) or not isinstance(p.get('objects'),list):raise ValueError('模型输出缺少 objects 数组')
 if len(p['objects'])>300:raise ValueError('单图目标数量超过 300')
 out={'objects':[],'unknown_regions':[],'image_quality':str(p.get('image_quality','unknown'))[:500]}
 for obj in p['objects']:
  cid=obj.get('class_id');b=obj.get('bbox');conf=obj.get('confidence',0)
  if type(cid) is not int or not 0<=cid<len(classes):raise ValueError(f'未知类别: {cid}')
  if not isinstance(b,list) or len(b)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in b):raise ValueError('检测框必须是四个有限数值')
  conf_raw=conf;conf=bounded_confidence(conf)
  if normalized:
   if not all(-.001<=v<=1000.001 for v in b):raise ValueError('坐标超出 0–1000')
   b=[max(0,min(1000,v)) for v in b]
   b=[b[0]*w/1000,b[1]*h/1000,b[2]*w/1000,b[3]*h/1000]
  if not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('检测框越界或面积为零')
  poly=obj.get('polygon',[])
  if poly and (not isinstance(poly,list) or len(poly)<3 or any(not isinstance(q,list) or len(q)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in q) or not (0<=q[0]<=w and 0<=q[1]<=h) for q in poly)):raise ValueError('分割多边形点必须在图内，至少三个点')
  out['objects'].append({'class_id':cid,'bbox':[round(v,2) for v in b],'confidence':float(conf),'polygon':poly,'text':str(obj.get('text',''))[:30],**({'color':obj['color']} if obj.get('color') in ['red','black','unknown'] else {}),**({'confidence_raw':conf_raw,'numeric_normalization':'float_roundoff_to_unit_interval'} if conf_raw!=conf else {})})
 for region in p.get('unknown_regions',[]):
  if not isinstance(region,dict) or 'bbox' not in region:raise ValueError('无法识别区域缺少 bbox')
  validated=validate_prediction({'objects':[{'class_id':0,'bbox':region['bbox'],'confidence':0}]},classes,w,h,normalized)
  out['unknown_regions'].append({'bbox':validated['objects'][0]['bbox'],'reason':str(region.get('reason','类别无法判定'))[:300]})
 return out

def compare(a,b,c):
 ao=a['objects'];bo=b['objects'];issues=[];matches=[]
 for role,p in [('generator',a),('reviewer',b)]:
  for region in p.get('unknown_regions',[]):issues.append({'type':'unreadable','message':f'{role} 发现无法判定类别的可见目标：'+region['reason'],'region':region['bbox']})
 if not ao and not bo and not c.get('allow_empty'):issues.append({'type':'empty','message':'两个模型均未检出目标；未启用空图自动通过'})
 if ao and bo:
  scores=np.array([[iou(x['bbox'],y['bbox']) for y in bo] for x in ao])
  rr,cc=linear_sum_assignment(1-scores)
  used_a=set();used_b=set()
  for ai,bi in zip(rr,cc):
   ai=int(ai);bi=int(bi);score=float(scores[ai,bi])
   if score<0.12:continue
   used_a.add(ai);used_b.add(bi);matches.append({'generator':ai,'reviewer':bi,'iou':round(score,4)})
   if ao[ai]['class_id']!=bo[bi]['class_id']:issues.append({'type':'class','message':f'目标 {ai+1} 类别或红黑颜色不一致','generator':ai,'reviewer':bi})
   if score<c['iou_threshold']:issues.append({'type':'location','message':f'目标 {ai+1} 定位差异较大（IoU {score:.2f}）','generator':ai,'reviewer':bi})
 else:used_a=set();used_b=set()
 for ai in range(len(ao)):
  if ai not in used_a:issues.append({'type':'unmatched_generator','message':f'生成模型目标 {ai+1} 未被复核模型独立检出','generator':ai})
 for bi in range(len(bo)):
  if bi not in used_b:issues.append({'type':'missed','message':f'复核模型额外检出目标 {bi+1}，可能漏标','reviewer':bi})
 for role,objs in [('generator',ao),('reviewer',bo)]:
  for idx,obj in enumerate(objs):
   if obj.get('color')=='unknown':issues.append({'type':'color_unknown','message':f'{role} 目标 {idx+1} 无法判断红黑颜色'})
   if obj.get('taxonomy_ambiguous'):issues.append({'type':'ambiguous_text','message':f'{role} 目标 {idx+1} 的识别文字涉及多个棋子类别'})
   if obj['confidence']<c['min_confidence']:issues.append({'type':'low_confidence','message':f'{role} 目标 {idx+1} 自报置信度不足（非校准概率）'})
   for j in range(idx):
    if iou(obj['bbox'],objs[j]['bbox'])>0.65:issues.append({'type':'duplicate','message':f'{role} 目标框高度重叠，可能重复标注'})
 return issues,matches

def circle_candidates(rgb):
 # Auxiliary evidence only. Never creates labels or establishes ground truth.
 h,w=rgb.shape[:2];scale=min(1,1200/max(h,w));im=cv2.resize(rgb,None,fx=scale,fy=scale);gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);gray=cv2.GaussianBlur(gray,(5,5),1.2)
 m=min(im.shape[:2]);cs=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=max(10,int(m*.045)),param1=110,param2=32,minRadius=max(7,int(m*.012)),maxRadius=max(10,int(m*.085)))
 if cs is None:return []
 # Reject accidental circles formed by board lines or disconnected text strokes.
 edges=cv2.Canny(gray,60,140);edges=cv2.dilate(edges,np.ones((3,3),np.uint8))
 candidates=[];angles=np.linspace(0,2*np.pi,96,endpoint=False)
 for x,y,r in cs[0]:
  xs=np.rint(x+r*np.cos(angles)).astype(int);ys=np.rint(y+r*np.sin(angles)).astype(int)
  inside=(xs>=0)&(xs<gray.shape[1])&(ys>=0)&(ys<gray.shape[0])
  if inside.mean()<.98:continue
  hits=(edges[ys[inside],xs[inside]]>0)
  if hits.mean()<.78:continue
  # Require support in every quadrant rather than just one strong arc.
  if min(np.array_split(hits,4)[q].mean() for q in range(4))<.6:continue
  candidates.append([round(float(x/scale),2),round(float(y/scale),2),round(float(r/scale),2)])
 return candidates[:100]

def audit_circles(objects,circles):
 issues=[]
 for x,y,r in circles:
  # A circle almost fully covered by any piece box does not indicate a miss.
  if not any(b['bbox'][0]-r*.3<=x<=b['bbox'][2]+r*.3 and b['bbox'][1]-r*.3<=y<=b['bbox'][3]+r*.3 for b in objects):
   issues.append({'type':'circle_uncovered','message':f'位置 ({x:.0f}, {y:.0f}) 存在未覆盖圆形候选，需要确认是否漏标','circle':[x,y,r]})
 return issues

def segment(rgb,objects):
 out=copy.deepcopy(objects);issues=[];h,w=rgb.shape[:2]
 cv2.setRNGSeed(7)
 for idx,o in enumerate(out):
  b=o['bbox'];x0=max(0,int(b[0]));y0=max(0,int(b[1]));x1=min(w,int(math.ceil(b[2])));y1=min(h,int(math.ceil(b[3])))
  bw=x1-x0;bh=y1-y0;pad=max(6,int(max(bw,bh)*.18));rx=max(0,x0-pad);ry=max(0,y0-pad);ex=min(w,x1+pad);ey=min(h,y1+pad)
  crop=rgb[ry:ey,rx:ex];mask=np.zeros(crop.shape[:2],np.uint8)
  try:
   if bw<8 or bh<8 or crop.shape[0]<10 or crop.shape[1]<10:raise ValueError('目标过小')
   mask[max(0,y0-ry):y1-ry,max(0,x0-rx):x1-rx]=cv2.GC_PR_FGD
   # A conservative central seed; actual contour still follows image pixels.
   sx=int((x0+x1)/2-rx);sy=int((y0+y1)/2-ry)
   cv2.ellipse(mask,(sx,sy),(max(1,int(bw*.20)),max(1,int(bh*.20))),0,0,360,cv2.GC_FGD,-1)
   cv2.grabCut(crop,mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),5,cv2.GC_INIT_WITH_MASK)
   fg=np.where((mask==1)|(mask==3),255,0).astype('uint8')
   contours,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   if not contours:raise ValueError('前景为空')
   cnt=max(contours,key=cv2.contourArea);area=cv2.contourArea(cnt);ratio=area/max(1,bw*bh)
   if ratio<.25 or ratio>1.3:raise ValueError(f'前景与框面积比异常 {ratio:.2f}')
   approx=cv2.approxPolyDP(cnt,max(.75,cv2.arcLength(cnt,True)*.003),True).reshape(-1,2)
   if len(approx)<5:raise ValueError('轮廓过于简单')
   pts=approx+np.array([rx,ry]);o['polygon']=[[int(x),int(y)] for x,y in pts];o['segmentation_method']='opencv_grabcut';o['mask_area']=round(area,2)
   px,py,pw,ph=cv2.boundingRect(cnt)
   # Touching expanded crop boundary indicates clipping or background leakage.
   if px<=0 or py<=0 or px+pw>=crop.shape[1] or py+ph>=crop.shape[0]:raise ValueError('前景接触扩展裁剪边界')
   o['segmentation_quality']='heuristic_pass'
  except (cv2.error,ValueError) as e:
   o['polygon']=[];o['segmentation_quality']='uncertain';issues.append({'type':'segmentation','message':f'目标 {idx+1} 轮廓提取不可靠：{str(e)[:100]}','generator':idx})
 return out,issues
