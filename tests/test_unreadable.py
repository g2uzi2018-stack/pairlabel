import copy,json,zipfile
import pytest
from fastapi.testclient import TestClient
from app import db
from app.main import app
from app.review import relabel,visual_edit
from app.classification import unreadable,assign_class,review_class_id
from app.annotation_formats import format_preview,yolo_text,coco_annotations
from app.exports import build_export
from test_review_controls import ready_item


def test_unknown_is_review_state_and_preserves_geometry_and_model_metadata():
    jid,iid,original=ready_item();classes=db.job(jid)['config']['classes']
    obj=original['objects'][0];before=copy.deepcopy(obj)
    result,changes,_=relabel(original,[-1],classes,[obj['class_id']])
    saved=result['objects'][0]
    assert unreadable(saved) and review_class_id(saved)==-1
    assert saved['class_id']==before['class_id']
    for key in ['bbox','polygon','confidence','segmentation_method','segmentation_score']:
        assert saved[key]==before[key]
    assert original['objects'][0]==before
    result,changes,_=relabel(result,[7],classes,[-1])
    assert not unreadable(result['objects'][0]) and result['objects'][0]['class_id']==7
    assert result['objects'][0]['color']=='black'


def test_unknown_api_save_remains_uncertain_and_resolve_becomes_approved():
    jid,iid,original=ready_item(failed=True);client=TestClient(app)
    before=db.rows('SELECT * FROM runs')
    r=client.post(f'/api/items/{iid}/review',json={'action':'labels','class_ids':[-1],'expected_class_ids':[0],'expected_revision':db.item(iid)['review_revision']})
    assert r.status_code==200
    saved=r.json();assert saved['status']=='uncertain' and not saved['recognition_failed']
    assert unreadable(saved['result']['objects'][0]) and db.rows('SELECT * FROM runs')==before
    assert saved['result']['objects'][0]['polygon']==original['objects'][0]['polygon']
    r=client.post(f'/api/items/{iid}/review',json={'action':'labels','class_ids':[7],'expected_class_ids':[-1],'expected_revision':saved['review_revision']})
    assert r.status_code==200 and r.json()['status']=='approved'
    assert not unreadable(r.json()['result']['objects'][0])


@pytest.mark.parametrize('fmt',['yolo','yolo-seg','coco'])
def test_unknown_previews_do_not_encode_the_historical_class(fmt):
    jid,iid,result=ready_item();classes=db.job(jid)['config']['classes']
    assign_class(result['objects'][0],-1,classes)
    r=format_preview(result['objects'],100,100,fmt,classes)
    assert not r['format_eligible'] and r['annotation']=='' and unreadable(r['objects'][0])
    with pytest.raises(ValueError,match='看不清'):yolo_text(result['objects'],100,100)
    with pytest.raises(ValueError,match='看不清'):coco_annotations(result['objects'])


def test_unknown_image_is_excluded_even_when_uncertain_exports_are_requested():
    jid,iid,result=ready_item();classes=db.job(jid)['config']['classes']
    assign_class(result['objects'][0],-1,classes)
    db.execute("UPDATE items SET status='uncertain',result=? WHERE id=?",(db.dumps(result),iid))
    _,other,other_result=ready_item()
    db.execute("UPDATE items SET job_id=?,status='approved' WHERE id=?",(jid,other))
    with zipfile.ZipFile(build_export(jid,'all',True,0)) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert [i['id'] for i in manifest['classification_excluded']]==[iid]
        assert [i['id'] for i in manifest['items']]==[other]
        assert not any(iid in name for name in z.namelist() if '/images/' in name or '/labels/' in name)


def test_visual_unknown_keeps_canonical_mask_and_advanced_edit_cannot_clear_it_accidentally():
    jid,iid,original=ready_item();classes=db.job(jid)['config']['classes'];o=original['objects'][0]
    edited,_,_=visual_edit(db.item(iid),[{'class_id':-1,'bbox':o['bbox'],'review_ref':{'kind':'result','index':0}}],classes,100,100,lambda _:None)
    assert unreadable(edited['objects'][0]) and edited['objects'][0]['polygon']==o['polygon']
    r=TestClient(app).post(f'/api/items/{iid}/review',json={'action':'edit','reason':'保留看不清标记','objects':edited['objects']})
    assert r.status_code==200 and r.json()['status']=='uncertain'
    assert unreadable(r.json()['result']['objects'][0])


def agent_request(iid,ids,revision=None,**extra):
    body={'class_ids':ids,'observations':[{'key':f'T{n}','text':'将' if cid==7 else '', 'reason':'字被遮挡'} for n,cid in enumerate(ids)],
          'expected_revision':revision or db.item(iid)['review_revision'],'batch':'test-luna',**extra}
    return TestClient(app).post(f'/api/items/{iid}/classification-review',json=body)


def test_agent_save_is_not_human_approval_preserves_geometry_and_is_idempotent():
    jid,iid,original=ready_item();runs=db.rows('SELECT * FROM runs')
    r=agent_request(iid,[7]);assert r.status_code==200
    saved=r.json();assert saved['status']=='accepted' and 'human_review' not in saved['result']
    assert saved['result']['objects'][0]['bbox']==original['objects'][0]['bbox']
    assert saved['result']['objects'][0]['polygon']==original['objects'][0]['polygon']
    assert db.rows('SELECT * FROM runs')==runs
    audits=db.rows("SELECT * FROM audit WHERE kind='agent_classification_review'");assert len(audits)==1
    assert agent_request(iid,[7]).status_code==200
    assert db.rows("SELECT * FROM audit WHERE kind='agent_classification_review'")==audits


def test_agent_unknown_blocks_training_and_cannot_overwrite_human_or_stale_results():
    jid,iid,_=ready_item();old=db.item(iid)['review_revision']
    r=agent_request(iid,[None]);assert r.status_code==200 and r.json()['status']=='uncertain'
    assert unreadable(r.json()['result']['objects'][0])
    # A different review batch with an obsolete version is rejected.
    assert agent_request(iid,[7],revision=old,batch='later').status_code==400
    db.execute("UPDATE items SET status='approved' WHERE id=?",(iid,))
    before=db.item(iid)
    assert agent_request(iid,[7],batch='other').status_code==400
    assert db.item(iid)==before


def test_agent_can_restore_only_own_existing_contours_and_never_invent_geometry():
    jid,iid,original=ready_item();db.save_run(iid,2,'reviewer','B',{'prediction':{'objects':original['objects']}})
    rid=db.one("SELECT id FROM runs WHERE item_id=? AND role='reviewer' AND attempt=2",(iid,))['id']
    db.execute('UPDATE items SET result=? WHERE id=?',(db.dumps({'objects':[]}),iid))
    r=agent_request(iid,[7],source_run_id=rid);assert r.status_code==200
    assert r.json()['result']['objects'][0]['polygon']==original['objects'][0]['polygon']
    assert r.json()['result']['classification_review']['recovered_existing_contour']
    _,other,_=ready_item();before=db.item(other)
    assert agent_request(other,[7],source_run_id=rid,batch='foreign').status_code==400
    assert db.item(other)==before


def test_agent_keeps_nonclassification_quality_issues():
    jid,iid,original=ready_item()
    kept=[{'type':kind,'message':kind} for kind in ['location','inventory','segmentation','mask_overlap','circle_uncovered','missed']]
    db.execute('UPDATE items SET issues=? WHERE id=?',(db.dumps([{'type':'class','message':'old class difference'},*kept]),iid))
    r=agent_request(iid,[7]);assert r.status_code==200
    assert r.json()['status']=='uncertain' and r.json()['issues']==kept
    assert r.json()['result']['issues']==kept


def test_agent_cannot_overwrite_partial_human_review_with_unreadable_target():
    jid,iid,original=ready_item();second=copy.deepcopy(original['objects'][0]);second['class_id']=1
    original['objects'].append(second)
    db.execute('UPDATE items SET result=? WHERE id=?',(db.dumps(original),iid))
    r=TestClient(app).post(f'/api/items/{iid}/review',json={'action':'labels','class_ids':[7,-1],'expected_class_ids':[0,1]})
    assert r.status_code==200 and r.json()['status']=='uncertain'
    before=db.item(iid)
    assert agent_request(iid,[0,0]).status_code==400
    assert db.item(iid)==before


def test_primary_reference_adjudication_has_explicit_provenance():
    jid,iid,original=ready_item()
    note={'key':'T1','text':'black_king','text_kind':'class_label_not_transcription','decision_source':'primary_reference_adjudication','luna_class_id':11,'reference_key':'reference-king'}
    r=agent_request(iid,[7],observations=[note]);assert r.status_code==200
    o=r.json()['result']['objects'][0]
    assert o['classification_source']=='codex_primary_reference'
    assert o['classification_review']['adjudicator']=='primary_agent'
    assert o['classification_review']['luna_class_id']==11
    assert o['classification_review']['text_kind']=='class_label_not_transcription'
    assert o['bbox']==original['objects'][0]['bbox'] and o['polygon']==original['objects'][0]['polygon']
