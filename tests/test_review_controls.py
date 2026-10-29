import hashlib
import json
import pytest
from fastapi.testclient import TestClient
from app import db
from app.config import DATA
from app.main import app
from PIL import Image
from test_quality import fixture_item, obj, pred


def ready_item(failed=False):
    jid, iid = fixture_item()
    image = DATA / db.item(iid)['path']
    pixels = Image.open(image).convert('RGB')
    pixels.putpixel((0, 0), tuple(bytes.fromhex(iid[:6])))
    pixels.save(image)
    db.execute('UPDATE items SET hash=? WHERE id=?', (hashlib.sha256(image.read_bytes()).hexdigest(), iid))
    geometry = {**obj(), 'polygon': [[10, 10], [70, 10], [65, 70]],
                'segmentation_method': 'sam2.1_hiera_large_cuda_fp32',
                'segmentation_score': .97, 'segmentation_quality': 'heuristic_pass',
                'mask_area': 3456, 'color': 'red', 'class_id_raw': 0}
    issues = [{'type': 'inference_error' if failed else 'class', 'message': 'original issue'}]
    result = {'objects': [geometry], 'issues': issues, 'attempt': 1}
    db.execute('UPDATE items SET status=?,attempt=1,result=?,issues=? WHERE id=?',
               ('uncertain', db.dumps(result), db.dumps(issues), iid))
    db.save_run(iid, 0, 'generator', 'A', {'prediction': pred(obj()), 'raw': 'immutable original'})
    return jid, iid, result


def test_label_dropdown_save_preserves_all_geometry_evidence_and_before_audit():
    jid, iid, original = ready_item()
    runs = db.rows('SELECT * FROM runs')
    client = TestClient(app)
    r = client.post(f'/api/items/{iid}/review', json={'action': 'labels', 'class_ids': [7], 'expected_class_ids': [0]})
    assert r.status_code == 200
    saved = r.json()
    assert saved['status'] == 'approved'
    target = saved['result']['objects'][0]
    assert target['class_id'] == 7 and target['color'] == 'black'
    for key, value in original['objects'][0].items():
        if key not in ('class_id', 'color'):
            assert target[key] == value
    assert db.rows('SELECT * FROM runs') == runs
    assert saved['issues'] == original['issues'] and saved['result']['issues'] == original['issues']
    audit = json.loads(db.one("SELECT detail FROM audit WHERE kind='human_review'")['detail'])
    assert audit['before'] == original
    assert audit['after']['objects'][0]['class_id'] == 7
    assert audit['after']['human_review']['label_changes'] == [{'index': 0, 'before': 0, 'after': 7}]
    assert saved['result']['human_review']['reason']
    assert client.get(f'/api/items/{iid}/format-preview').json()['included']


def test_one_click_confirmation_without_changes_records_review_without_rewriting_objects():
    _, iid, original = ready_item()
    r = TestClient(app).post(f'/api/items/{iid}/review', json={'action': 'labels', 'class_ids': [0]})
    assert r.status_code == 200
    result = r.json()['result']
    assert result['objects'] == original['objects']
    assert result['human_review']['label_changes'] == []


@pytest.mark.parametrize('payload,code', [
    ({'class_ids': []}, 400), ({'class_ids': [99]}, 400),
    ({'class_ids': [True]}, 422), ({'class_ids': ['7']}, 422),
    ({'class_ids': [7], 'expected_class_ids': [3]}, 400),
])
def test_invalid_or_stale_label_edits_do_not_write(payload, code):
    _, iid, _ = ready_item()
    before = db.rows('SELECT * FROM items')
    r = TestClient(app).post(f'/api/items/{iid}/review', json={'action': 'labels', **payload})
    assert r.status_code == code
    assert db.rows('SELECT * FROM items') == before
    assert not db.rows("SELECT * FROM audit WHERE kind='human_review'")


def test_empty_failed_detection_cannot_be_accidentally_approved_as_negative_by_labels():
    _, iid, _ = ready_item(failed=True)
    db.execute('UPDATE items SET result=? WHERE id=?', (db.dumps({'objects': []}), iid))
    r = TestClient(app).post(f'/api/items/{iid}/review', json={'action': 'labels', 'class_ids': []})
    assert r.status_code == 400 and db.item(iid)['status'] == 'uncertain'


def test_failed_filter_separates_model_failure_from_semantic_uncertainty_and_clears_after_review():
    jid, failed, _ = ready_item(failed=True)
    # Put a second, semantic uncertainty into the same task without touching models.
    other_jid, normal, _ = ready_item()
    db.execute('UPDATE items SET job_id=? WHERE id=?', (jid, normal))
    client = TestClient(app)
    assert client.get(f'/api/jobs/{jid}').json()['review_counts'] == {'failed': 1, 'uncertain': 1}
    failures = client.get(f'/api/jobs/{jid}/items?status=failed').json()
    uncertain = client.get(f'/api/jobs/{jid}/items?status=uncertain').json()
    assert [i['id'] for i in failures['items']] == [failed]
    assert failures['items'][0]['recognition_failed']
    assert [i['id'] for i in uncertain['items']] == [normal]
    assert client.post(f'/api/items/{failed}/review', json={'action': 'labels', 'class_ids': [0]}).status_code == 200
    assert client.get(f'/api/jobs/{jid}/items?status=failed').json()['total'] == 0


def test_failed_retry_does_not_queue_semantic_uncertainty_or_resolved_historical_errors():
    jid, failed, _ = ready_item(failed=True)
    _, normal, _ = ready_item()
    db.execute('UPDATE items SET job_id=? WHERE id=?', (jid, normal))
    _, approved, _ = ready_item(failed=True)
    db.execute("UPDATE items SET job_id=?,status='approved' WHERE id=?", (jid, approved))
    client = TestClient(app)
    evidence = db.rows('SELECT * FROM runs')
    r = client.post(f'/api/jobs/{jid}/action', json={'action': 'retry_failed'})
    assert r.status_code == 200
    assert db.item(failed)['status'] == 'pending' and db.item(failed)['attempt'] == 2
    assert db.item(normal)['status'] == 'uncertain' and db.item(normal)['attempt'] == 1
    assert db.item(approved)['status'] == 'approved'
    assert db.rows('SELECT * FROM runs') == evidence


def test_active_task_allows_completed_image_review_but_blocks_retry_and_pending_edits():
    jid, iid, _ = ready_item(failed=True)
    db.set_job(jid, 'running')
    c = TestClient(app)
    assert c.post(f'/api/items/{iid}/review', json={'action': 'labels', 'class_ids': [0]}).status_code == 200
    assert c.post(f'/api/jobs/{jid}/action', json={'action': 'retry_failed'}).status_code == 400
    db.execute("UPDATE items SET status='pending' WHERE id=?", (iid,))
    assert c.post(f'/api/items/{iid}/review', json={'action': 'labels', 'class_ids': [0]}).status_code == 400


def test_advanced_review_still_requires_reason():
    _, iid, _ = ready_item()
    assert TestClient(app).post(f'/api/items/{iid}/review', json={'action': 'approve'}).status_code == 400
