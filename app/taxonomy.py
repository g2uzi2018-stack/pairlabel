"""Deterministic glyph-to-kind mapping; never infer color from a glyph."""
import copy
KINDS={'king':'帅帥将將','advisor':'士仕','elephant':'相象','horse':'马馬傌','rook':'车車俥','cannon':'炮砲','pawn':'兵卒'}

def normalize_taxonomy(prediction,classes,enabled=True):
 p=copy.deepcopy(prediction);adjustments=[]
 if not enabled or not all(c.get('piece') and c.get('color') for c in classes):return p,adjustments
 for idx,o in enumerate(p['objects']):
  kinds={kind for kind,chars in KINDS.items() if any(ch in o.get('text','') for ch in chars)}
  if len(kinds)>1:
   o['taxonomy_ambiguous']=True;o.setdefault('confidence_raw',o['confidence']);o['confidence']=0;continue
  if len(kinds)!=1:continue
  kind=next(iter(kinds));declared=classes[o['class_id']]
  color=o.get('color') if o.get('color') in ['red','black'] else declared['color']
  if o.get('color')=='unknown':
   o['confidence_raw']=o['confidence'];o['confidence']=0;continue
  if declared['piece']==kind and declared['color']==color:continue
  candidates=[c for c in classes if c['piece']==kind and c['color']==color]
  if len(candidates)!=1:continue
  original=o['class_id'];o.setdefault('class_id_raw',original);o['class_id']=candidates[0]['id'];o['taxonomy_normalized']=True
  adjustments.append({'object':idx,'original_class_id':original,'class_id':o['class_id'],'text':o.get('text'),'reason':'按本模型实际识别文字映射棋子类别；保留本模型独立判断的颜色，未使用另一模型结果'})
 return p,adjustments
