import copy,json,zipfile
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
from app import db
from app.config import DEFAULT,DATA,CLASSES
from app.geometry import compare,validate_prediction,segment,iou
from app.exports import build_export
from app.pipeline import Worker
from app.main import import_image,manual_review,Review

def obj(cid=0,b=None,confidence=.95):return {'class_id':cid,'bbox':b or [10,10,70,70],'confidence':confidence,'polygon':[],'text':'帅'}
def pred(*objects):return {'objects':list(objects),'image_quality':'clear'}
def config():
 c=copy.deepcopy(DEFAULT);c.update(classes=copy.deepcopy(CLASSES),annotation_profile='xiangqi',segmentation=False,circle_audit=False,max_attempts=2);return c

def fixture_item(c=None):
 c=c or config();jid=db.new_job('验证任务',c);(DATA/'jobs'/jid/'results').mkdir(parents=True,exist_ok=True)
 im=Image.new('RGB',(100,100),'white');import io
 b=io.BytesIO();im.save(b,format='PNG');import_image(jid,b.getvalue(),'测试.png');iid=db.one('SELECT id FROM items WHERE job_id=?',(jid,))['id'];db.set_job(jid,'paused');return jid,iid

def test_independent_extra_detection_and_color_disagreement_are_not_passed():
 issues,matches=compare(pred(obj()),pred(obj(7),obj(6,[80,10,95,30])),config());types={x['type'] for x in issues}
 assert {'class','missed'}<=types and len(matches)==1

def test_empty_consensus_is_conservative():
 assert compare(pred(),pred(),config())[0][0]['type']=='empty'
 c=config();c['allow_empty']=True;assert compare(pred(),pred(),c)[0]==[]

def test_duplicate_and_uncertain_boxes_are_flagged():
 p=pred(obj(confidence=.4),obj());types={x['type'] for x in compare(p,p,config())[0]};assert {'duplicate','low_confidence'}<=types

@pytest.mark.parametrize('o',[{'class_id':100,'bbox':[0,0,10,10],'confidence':1},{'class_id':0,'bbox':[100,100,0,0],'confidence':1},{'class_id':0,'bbox':[0,0,float('nan'),100],'confidence':1}])
def test_malformed_model_output_fails_closed(o):
 with pytest.raises(ValueError):validate_prediction(pred(o),CLASSES,100,100)

def test_retry_history_and_failed_reviewer_never_auto_accept():
 jid,iid=fixture_item();c=config();db.save_run(iid,0,'generator','A',{'prediction':pred(obj()),'raw':'A'});db.save_run(iid,0,'reviewer','B',error='output truncated');w=Worker();w.finalize(jid,iid,0,c)
 i=db.item(iid);assert i['status']=='pending' and i['attempt']==1
 assert db.one('SELECT count(*) n FROM runs WHERE item_id=?',(iid,))['n']==2
 db.save_run(iid,1,'generator','A',{'prediction':pred(obj()),'raw':'A2'});db.save_run(iid,1,'reviewer','B',error='invalid JSON');w.finalize(jid,iid,1,c);assert db.item(iid)['status']=='uncertain'
 assert len(list((DATA/'jobs'/jid/'results').glob('*.json')))==2

def test_resume_reuses_durable_independent_predictions():
 jid,iid=fixture_item();c=config();c['max_attempts']=1;db.set_job(jid,'running');db.save_run(iid,0,'generator','A',{'prediction':pred(obj()),'raw':'preserved'})
 class Fake:
  def __init__(self):self.calls=[]
  def infer(self,role,*args):self.calls.append(role);return {'prediction':pred(obj()),'raw':'B'}
 w=Worker();w.manager=Fake();w.process(jid,c)
 assert w.manager.calls==['reviewer'];assert db.item(iid)['status']=='accepted';assert db.job(jid)['status']=='completed'

def test_export_uses_only_eligible_images_and_valid_coordinates():
 jid,iid=fixture_item();o=obj();o['polygon']=[[10,10],[70,10],[70,70],[10,70]];db.execute('UPDATE items SET status=?,result=? WHERE id=?',('accepted',db.dumps(pred(o)),iid));p=build_export(jid,'all',False,0)
 with zipfile.ZipFile(p) as z:
  det=z.read(f'yolo/labels/train/{iid}.txt').decode().split();assert det==['0','0.4000000','0.4000000','0.6000000','0.6000000']
  coco=json.loads(z.read('coco/annotations/instances_train.json'));assert coco['annotations'][0]['bbox']==[10,10,60,60];assert coco['annotations'][0]['category_id']==1
  assert len(z.read(f'yolo-seg/labels/train/{iid}.txt').decode().split())==9
  assert 'review/audit.jsonl' in z.namelist()
  import yaml
  spec=yaml.safe_load(z.read('yolo/dataset.yaml'));assert 'path' not in spec and spec['train']=='images/train' and spec['names'][0]=='red_king'
 db.execute("UPDATE items SET status='uncertain' WHERE id=?",(iid,))
 with pytest.raises(ValueError,match='没有符合条件'):build_export(jid)
 assert build_export(jid,include_uncertain=True).exists()

def test_segmentation_missing_excludes_entire_image():
 jid,iid=fixture_item();db.execute('UPDATE items SET status=?,result=? WHERE id=?',('accepted',db.dumps(pred(obj())),iid));p=build_export(jid,'all',False,0)
 with zipfile.ZipFile(p) as z:
  assert f'yolo/images/train/{iid}.png' in z.namelist();assert f'yolo-seg/images/train/{iid}.png' not in z.namelist();assert json.loads(z.read('manifest.json'))['segmentation_excluded']

def test_human_edit_preserves_model_and_original_issues():
 jid,iid=fixture_item();db.execute('UPDATE items SET status=?,result=?,issues=? WHERE id=?',('uncertain',db.dumps(pred(obj())),db.dumps([{'type':'class','message':'不一致'}]),iid));db.save_run(iid,0,'generator','A',{'prediction':pred(obj()),'raw':'raw-independent'})
 manual_review(iid,Review(action='edit',reason='核对红帅边缘与文字',objects=[obj(7)]));i=db.item(iid);assert i['status']=='approved' and i['result']['objects'][0]['class_id']==7 and i['issues']
 assert db.one('SELECT raw FROM runs WHERE item_id=?',(iid,))['raw']=='raw-independent'

def test_actual_pixel_segmentation_not_rectangle():
 import cv2
 rgb=np.full((160,160,3),[155,125,80],dtype='uint8');cv2.circle(rgb,(80,80),35,(245,222,167),-1)
 objects,issues=segment(rgb,[obj(b=[44,44,116,116])]);assert not issues;poly=objects[0]['polygon'];assert len(poly)>8;area=abs(cv2.contourArea(np.array(poly,dtype='int32')));assert .65<area/(72*72)<.85


def test_unreadable_piece_prevents_auto_accept_even_if_known_objects_agree():
 a=validate_prediction({'objects':[{'class_id':0,'bbox':[100,100,700,700],'confidence':.99}], 'unknown_regions':[{'bbox':[800,100,950,250],'reason':'背面，文字不可见'}]},CLASSES,100,100)
 issues,_=compare(a,a,config());assert any(i['type']=='unreadable' for i in issues)

def test_board_intersections_do_not_become_missing_piece_evidence():
 import cv2
 from app.geometry import circle_candidates
 rgb=np.full((500,500,3),220,dtype='uint8')
 for x in range(30,490,45):cv2.line(rgb,(x,20),(x,480),(90,90,90),2)
 for y in range(20,490,45):cv2.line(rgb,(30,y),(480,y),(90,90,90),2)
 assert circle_candidates(rgb)==[]


def test_matching_unreadable_backs_are_retained_without_futile_auto_retry():
 jid,iid=fixture_item();p=pred(obj());p['unknown_regions']=[{'bbox':[75,75,95,95],'reason':'背面朝上，无字'}]
 for role in ['generator','reviewer']:db.save_run(iid,0,role,role,{'prediction':p})
 Worker().finalize(jid,iid,0,config());i=db.item(iid);assert i['status']=='uncertain' and i['attempt']==0 and i['result']['retry_skipped']


def test_glyph_mapping_corrects_kind_and_preserves_declared_color_and_raw_evidence():
 from app.taxonomy import normalize_taxonomy
 p=pred(obj(1),obj(8));p['objects'][0]['text']='帥';p['objects'][1]['text']='帥'
 out,changes=normalize_taxonomy(p,CLASSES)
 assert [x['class_id'] for x in out['objects']]==[0,7]
 assert [x['class_id_raw'] for x in out['objects']]==[1,8]
 assert [x['class_id'] for x in p['objects']]==[1,8] and len(changes)==2


def test_crop_review_refuses_proposals_that_lose_previous_objects():
 from app.crop_review import eligible
 jid,iid=fixture_item();i=db.item(iid);props=DATA/'jobs'/jid/'results'/f'{iid}.proposals.json';props.write_text(db.dumps({'objects':[obj(b=[80,80,95,95])]}))
 db.save_run(iid,0,'generator','A',{'prediction':pred(obj(b=[0,0,10,10]))})
 assert eligible(DATA/i['path'],'generator',1,config()) is None

def test_crop_vote_distinguishes_ui_coin_from_chess_piece_and_records_shared_geometry(monkeypatch):
 from app import crop_review
 jid,iid=fixture_item();i=db.item(iid);c=config();c['annotation_instructions']='只标注桌面上的实体，排除界面图标'
 class Fake:
  progress={}
  def ensure(self,*args):pass
 m=Fake()
 response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'items':[{'patch':1,'kind':'piece','class_id':1,'text':'帥','color':'red','confidence':.98,'reason':''},{'patch':2,'kind':'not_piece','class_id':-1,'text':'分','color':'unknown','confidence':.99,'reason':'界面积分图标'}]})}}]}
 def respond(url,payload,*args):
  from app.prompts import build_crop_prompt
  assert payload['messages'][0]['content'][1]['text']==build_crop_prompt(c,2)
  assert c['annotation_instructions'] in payload['messages'][0]['content'][1]['text']
  scores=payload['response_format']['json_schema']['schema']['properties']['items']['items']['properties']['confidence']['enum']
  assert 4 not in scores and .98 in scores and min(scores)==0 and max(scores)==1
  return response
 monkeypatch.setattr(crop_review,'http_json',respond)
 results=[crop_review.classify(m,role,DATA/i['path'],c,1,({'objects':[obj(),obj(b=[75,75,95,95])]},pred(obj()))) for role in ['generator','reviewer']]
 for result in results:
  p=result['prediction'];assert p['objects'][0]['class_id']==0 and p['objects'][0]['class_id_raw']==1
  assert len(p['rejected_proposals'])==1 and p['spatial_source']=='sam2_pixel_proposals'
  assert result['metrics']['independent_classification']
 assert results[0]['metrics']['pages'][0]['input_sha256']==results[1]['metrics']['pages'][0]['input_sha256']
 assert results[0]['metrics']['pages'][0]['prompt']==results[1]['metrics']['pages'][0]['prompt']


def test_explicit_independent_color_takes_priority_over_enumerated_id():
 from app.taxonomy import normalize_taxonomy
 p=pred(obj(1));p['objects'][0].update(text='將',color='black')
 out,_=normalize_taxonomy(p,CLASSES);assert out['objects'][0]['class_id']==7 and out['objects'][0]['class_id_raw']==1


def test_back_without_peer_readable_detection_is_retained_but_peer_readable_overlap_retries():
 jid,iid=fixture_item();a=pred(obj());a['unknown_regions']=[{'bbox':[75,75,95,95],'reason':'背面朝上，无字'}]
 db.save_run(iid,0,'generator','A',{'prediction':a});db.save_run(iid,0,'reviewer','B',{'prediction':pred(obj())});Worker().finalize(jid,iid,0,config());assert db.item(iid)['status']=='uncertain'
 db.execute("UPDATE items SET status='pending',attempt=0 WHERE id=?",(iid,));db.save_run(iid,0,'reviewer','B',{'prediction':pred(obj(),obj(7,[75,75,95,95]))});Worker().finalize(jid,iid,0,config());assert db.item(iid)['status']=='pending' and db.item(iid)['attempt']==1


def test_runtime_float_roundoff_at_limits_is_normalized_but_real_out_of_range_is_rejected():
 from app.geometry import bounded_confidence
 assert bounded_confidence(1.0000001192092896)==1
 with pytest.raises(ValueError):bounded_confidence(1.01)
 o={'class_id':0,'bbox':[0,0,1000.0000610351562,1000],'confidence':1.0000001192092896}
 p=validate_prediction(pred(o),CLASSES,100,100);assert p['objects'][0]['bbox']==[0,0,100,100] and p['objects'][0]['confidence_raw']>1


def test_export_api_returns_local_path_and_streams_existing_archive():
 from fastapi.testclient import TestClient
 from app.main import app
 jid,iid=fixture_item();db.execute('UPDATE items SET status=?,result=? WHERE id=?',('accepted',db.dumps(pred(obj())),iid))
 client=TestClient(app);r=client.post(f'/api/jobs/{jid}/export',json={'format':'yolo','val_ratio':0})
 assert r.status_code==200 and Path(r.json()['server_path']).exists()
 downloaded=client.get(r.json()['download_url']);assert downloaded.status_code==200 and downloaded.content[:2]==b'PK'
 assert 'attachment' in downloaded.headers['content-disposition']


def test_unknown_color_cannot_pass_by_lowering_confidence_threshold():
 c=config();c['min_confidence']=0;p=pred(obj());p['objects'][0]['color']='unknown'
 assert any(x['type']=='color_unknown' for x in compare(p,p,c)[0])


def test_failed_sam_mask_keeps_original_box_and_retains_uncertainty(monkeypatch,tmp_path):
 import sys,types,contextlib
 from app.segmentation import Segmenter
 monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(inference_mode=contextlib.nullcontext))
 monkeypatch.setitem(sys.modules,'sam2.build_sam',types.SimpleNamespace(build_sam2=None))
 monkeypatch.setitem(sys.modules,'sam2.sam2_image_predictor',types.SimpleNamespace(SAM2ImagePredictor=None))
 class FailedPredictor:
  def set_image(self,*args):pass
  def predict(self,**kwargs):raise RuntimeError('mask inference failed')
 checkpoint=tmp_path/'weights.pt';checkpoint.touch();s=Segmenter();s.predictor=FailedPredictor();s.signature=str(checkpoint.resolve())
 c=config();c['sam2']={'weights':str(checkpoint)};source=obj(b=[15,20,55,65])
 out,issues=s.segment(np.zeros((100,100,3),dtype='uint8'),[source],c)
 assert out[0]['bbox']==source['bbox'] and out[0]['polygon']==[] and out[0]['segmentation_quality']=='uncertain' and issues


def test_mask_refinement_cannot_merge_two_agreed_objects_and_still_auto_accept():
 c=config();c.update(segmentation=True,segmentation_backend='grabcut',max_attempts=1)
 jid,iid=fixture_item(c);p=pred(obj(0,[10,10,40,40]),obj(7,[60,60,90,90]))
 for role in ['generator','reviewer']:db.save_run(iid,0,role,role,{'prediction':p})
 class SnappedMasks:
  def segment(self,rgb,objects,settings):
   out=copy.deepcopy(objects)
   for o in out:o.update(bbox=[10,10,40,40],polygon=[[10,10],[40,10],[40,40],[10,40]])
   return out,[]
 w=Worker();w.segmenter=SnappedMasks();w.finalize(jid,iid,0,c);i=db.item(iid)
 assert i['status']=='uncertain' and any(x['type']=='mask_overlap' for x in i['issues'])


@pytest.mark.parametrize('profile',['general','xiangqi'])
@pytest.mark.parametrize('attempt',[0,1])
def test_both_models_receive_identical_custom_task_pixels_prompt_and_schema(monkeypatch,profile,attempt):
 from app import inference
 from app.prompts import build_detection_prompt
 c=copy.deepcopy(DEFAULT)
 c.update(annotation_profile=profile,classes=[{'id':0,'name':'sprocket','label':'链轮'},{'id':1,'name':'sensor','label':'传感器'}],annotation_instructions='只标注托盘内的完整实体；传感器无文字时也按外观识别。')
 for entry in c['classes']:entry.update(piece='king',color='red')
 calls=[]
 def respond(url,payload,**kwargs):
  calls.append(copy.deepcopy(payload))
  o={'class_id':1,'bbox':[100,200,800,900],'confidence':.96,'text':''}
  if profile=='xiangqi':o['color']='red'
  return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'objects':[o],'unknown_regions':[],'image_quality':'clear'})}}]}
 monkeypatch.setattr(inference,'http_json',respond)
 manager=inference.ModelManager();monkeypatch.setattr(manager,'ensure',lambda *args:None)
 outs=[manager._infer_single(role,'unused.png',c,attempt,Image.new('RGB',(120,80),'white')) for role in ['generator','reviewer']]
 assert len(calls)==2 and calls[0]['messages']==calls[1]['messages']
 assert calls[0]['response_format']==calls[1]['response_format']
 assert calls[0]['model']!=calls[1]['model']
 prompt=calls[0]['messages'][0]['content'][1]['text']
 assert prompt==build_detection_prompt(c,attempt) and c['annotation_instructions'] in prompt
 assert '0: 链轮 (sprocket)' in prompt and '1: 传感器 (sensor)' in prompt
 properties=calls[0]['response_format']['json_schema']['schema']['properties']['objects']['items']['properties']
 assert ('color' in properties)==(profile=='xiangqi')
 assert properties['confidence']['enum']==[i/100 for i in range(101)]
 for out in outs:
  assert out['prediction']['objects'][0]['text']==''
  assert out['prediction']['objects'][0]['bbox']==[12,16,96,72]
  assert out['metrics']['prompt']==prompt and out['raw']
 assert outs[0]['metrics']['prompt_sha256']==outs[1]['metrics']['prompt_sha256']


def test_explicit_general_with_chess_metadata_never_applies_chess_rules(monkeypatch):
 from app import pipeline,crop_review,inference
 c=config();c.update(annotation_profile='general',circle_audit=True,segmentation=True,segmentation_backend='sam2')
 jid,iid=fixture_item(c);i=db.item(iid)
 p=pred(obj(0,[5,5,35,35]),obj(0,[55,55,85,85]))
 for o in p['objects']:o.update(text='車',color='unknown',taxonomy_ambiguous=True)
 for role in ['generator','reviewer']:db.save_run(iid,0,role,role,{'prediction':p,'raw':f'original-{role}'})
 def forbidden(*args,**kwargs):pytest.fail('general mode entered a chess-specific path')
 monkeypatch.setattr(pipeline,'circle_candidates',forbidden);monkeypatch.setattr(pipeline,'audit_circles',forbidden)
 monkeypatch.setattr(crop_review,'classify',forbidden)
 class Masks:
  def __init__(self):self.calls=0
  def segment(self,rgb,objects,settings):
   self.calls+=1
   return copy.deepcopy(objects),[]
 w=Worker();w.segmenter=Masks();w.finalize(jid,iid,0,c);out=db.item(iid)
 assert out['status']=='accepted' and out['issues']==[]
 assert [o['class_id'] for o in out['result']['objects']]==[0,0]
 assert out['result']['taxonomy_adjustments']=={'generator':[],'reviewer':[]}
 assert out['result']['objects'][0]['color']=='unknown' and out['result']['objects'][0]['confidence']==.95
 assert w.segmenter.calls==1 and not (DATA/'jobs'/jid/'results'/f'{iid}.proposals.json').exists()
 # Even an existing chess proposal manifest must not enable the crop strategy.
 (DATA/'jobs'/jid/'results'/f'{iid}.proposals.json').write_text(db.dumps({'objects':p['objects']}))
 assert crop_review.eligible(DATA/i['path'],'generator',1,c) is None
 manager=inference.ModelManager();calls=[]
 def single(role,path,settings,attempt,image=None):
  calls.append(image.size)
  return {'prediction':{'objects':[],'unknown_regions':[]},'raw':'independent tile','metrics':{}}
 monkeypatch.setattr(manager,'_infer_single',single)
 retried=manager.infer('generator',DATA/i['path'],c,1)
 assert len(calls)==4 and retried['metrics']['retry_strategy']=='overlapping_2x2_original_image'
 report=json.loads((DATA/'jobs'/jid/'results'/f'{iid}.attempt-0.json').read_text())
 assert {r['raw'] for r in report['runs']}=={'original-generator','original-reviewer'}


def test_general_back_or_no_text_does_not_skip_independent_retry():
 c=copy.deepcopy(DEFAULT);c.update(segmentation=False,max_attempts=2)
 jid,iid=fixture_item(c);p=pred(obj());p['objects'][0]['text']=''
 p['unknown_regions']=[{'bbox':[75,75,95,95],'reason':'背面朝上，无字，外观模糊'}]
 for role in ['generator','reviewer']:db.save_run(iid,0,role,role,{'prediction':p,'raw':role})
 Worker().finalize(jid,iid,0,c);i=db.item(iid)
 assert i['status']=='pending' and i['attempt']==1 and 'retry_skipped' not in i['result']
 assert any(issue['type']=='unreadable' for issue in i['issues'])
 assert 'circle_candidates' not in i['result']


def test_general_mask_failures_and_final_overlap_remain_uncertain():
 c=copy.deepcopy(DEFAULT);c.update(segmentation=True,max_attempts=1)
 jid,iid=fixture_item(c);p=pred(obj(0,[10,10,40,40]),obj(1,[60,60,90,90]))
 for role in ['generator','reviewer']:db.save_run(iid,0,role,role,{'prediction':p})
 class UnreliableMasks:
  def segment(self,rgb,objects,settings):
   out=copy.deepcopy(objects)
   for o in out:o.update(bbox=[10,10,40,40],segmentation_quality='uncertain',polygon=[])
   return out,[{'type':'segmentation','message':'mask unreliable'}]
 w=Worker();w.segmenter=UnreliableMasks();w.finalize(jid,iid,0,c);i=db.item(iid)
 assert i['status']=='uncertain' and {'segmentation','mask_overlap'}<={issue['type'] for issue in i['issues']}
