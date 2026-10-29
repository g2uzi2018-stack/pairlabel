import json,threading
import pytest
from fastapi.testclient import TestClient
from app import db
from app.main import app
from app.pipeline import Worker
from test_quality import config,obj,pred
from test_review_controls import ready_item

def visual_request(iid,objects,revision=None):
    return TestClient(app).post(f'/api/items/{iid}/review',json={'action':'visual','objects':objects,'expected_revision':revision or db.item(iid)['review_revision']})

def test_adopt_reviewer_restores_empty_result_with_canonical_mask_and_audit():
    _,iid,_=ready_item()
    db.execute('UPDATE items SET result=? WHERE id=?',(db.dumps({'objects':[],'attempt':1}),iid))
    candidate={**obj(),'polygon':[[10,10],[70,10],[65,70]],'segmentation_score':.97,'proposal_index':0}
    db.save_run(iid,1,'reviewer','B',{'prediction':pred(candidate)})
    rid=db.one("SELECT id FROM runs WHERE item_id=? AND role='reviewer'",(iid,))['id']
    before=db.rows('SELECT * FROM runs')
    r=visual_request(iid,[{'class_id':7,'bbox':candidate['bbox'],'review_ref':{'kind':'run','run_id':rid,'index':0}}])
    assert r.status_code==200
    i=r.json();o=i['result']['objects'][0]
    assert i['status']=='approved' and o['class_id']==7 and o['color']=='black'
    assert o['bbox']==candidate['bbox'] and o['polygon']==candidate['polygon']
    assert o['segmentation_score']==.97 and o['human_source_run']==rid
    assert db.rows('SELECT * FROM runs')==before
    audit=json.loads(db.one("SELECT detail FROM audit WHERE kind='human_review'")['detail'])
    assert audit['before']['objects']==[] and audit['after']['objects'][0]['class_id']==7
    assert TestClient(app).get(f'/api/items/{iid}/format-preview?format=yolo-seg').json()['included']

def test_manual_box_is_not_presented_as_a_segmentation_mask_or_model_confidence():
    _,iid,_=ready_item()
    r=visual_request(iid,[{'class_id':0,'bbox':[12,15,60,80],'polygon':[[1,1],[2,2],[3,3]],'segmentation_score':1,'review_ref':{'kind':'drawn'}}])
    assert r.status_code==200
    o=r.json()['result']['objects'][0]
    assert o['bbox']==[12,15,60,80] and o['polygon']==[] and o['confidence']==0
    assert 'segmentation_score' not in o and o['annotation_source']=='human_box'
    c=TestClient(app)
    assert c.get(f'/api/items/{iid}/format-preview?format=yolo').json()['included']
    assert not c.get(f'/api/items/{iid}/format-preview?format=yolo-seg').json()['included']

def test_existing_object_edit_keeps_geometry_metadata_and_records_removal():
    _,iid,original=ready_item()
    o=original['objects'][0]
    r=visual_request(iid,[{'class_id':7,'bbox':o['bbox'],'review_ref':{'kind':'result','index':0}}])
    assert r.status_code==200
    saved=r.json()['result']['objects'][0]
    assert saved['polygon']==o['polygon'] and saved['segmentation_method']==o['segmentation_method']
    assert r.json()['result']['human_review']['object_sources']==[{'kind':'result','index':0}]

@pytest.mark.parametrize('edits',[
    [],[{'class_id':0,'bbox':[-1,0,20,20],'review_ref':{'kind':'drawn'}}],
    [{'class_id':999,'bbox':[1,1,20,20],'review_ref':{'kind':'drawn'}}],
    [{'class_id':0,'bbox':[1,1,20,20],'review_ref':{'kind':'result','index':0}}],
    [{'class_id':0,'bbox':[10,10,70,70],'review_ref':{'kind':'result','index':True}}],
])
def test_invalid_visual_edits_do_not_change_items(edits):
    _,iid,_=ready_item();before=db.rows('SELECT * FROM items')
    assert visual_request(iid,edits).status_code==400
    assert db.rows('SELECT * FROM items')==before

def test_model_result_from_another_image_cannot_be_adopted():
    _,iid,_=ready_item();_,other,_=ready_item()
    rid=db.one('SELECT id FROM runs WHERE item_id=?',(other,))['id']
    assert visual_request(iid,[{'class_id':0,'bbox':[10,10,70,70],'review_ref':{'kind':'run','run_id':rid,'index':0}}]).status_code==400

def test_stale_draft_cannot_replace_newer_result_even_when_class_ids_match():
    _,iid,_=ready_item();revision=db.item(iid)['review_revision']
    result=db.item(iid)['result'];result['objects'][0]['bbox']=[11,10,70,70]
    db.execute('UPDATE items SET result=? WHERE id=?',(db.dumps(result),iid))
    r=visual_request(iid,[{'class_id':0,'bbox':[12,15,60,80],'review_ref':{'kind':'drawn'}}],revision)
    assert r.status_code==400 and db.item(iid)['result']==result

def test_background_worker_and_other_image_manual_review_do_not_overwrite_each_other():
    jid,running,_=ready_item();_,other,_=ready_item()
    db.execute('UPDATE items SET job_id=? WHERE id=?',(jid,other))
    db.execute("UPDATE items SET status='pending',attempt=0 WHERE id=?",(running,))
    db.set_job(jid,'running');started=threading.Event();release=threading.Event();failures=[]
    w=Worker();c=config();c['max_attempts']=1
    def infer(*args):
        started.set();assert release.wait(5);return {'prediction':pred(obj()),'raw':'worker output'}
    w.manager.infer=infer
    def process():
        try:w.process(jid,c)
        except Exception as e:failures.append(e)
    thread=threading.Thread(target=process);thread.start()
    try:
        assert started.wait(3)
        client=TestClient(app)
        assert client.post(f'/api/items/{other}/review',json={'action':'labels','class_ids':[7],'expected_revision':db.item(other)['review_revision']}).status_code==200
        assert client.post(f'/api/items/{running}/review',json={'action':'labels','class_ids':[0]}).status_code==400
    finally:release.set();thread.join(5)
    assert not thread.is_alive() and not failures
    assert db.item(other)['status']=='approved' and db.item(other)['result']['objects'][0]['class_id']==7
    assert db.item(running)['status']=='accepted'

def test_item_list_and_detail_share_the_same_review_revision():
    jid,iid,_=ready_item();client=TestClient(app)
    assert client.get(f'/api/items/{iid}').json()['review_revision']==client.get(f'/api/jobs/{jid}/items').json()['items'][0]['review_revision']
