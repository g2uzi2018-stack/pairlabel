import json
import sqlite3
import zipfile
import pytest
from fastapi.testclient import TestClient
from app import db
from app.config import DATA
from app.exports import build_export
from app.main import app
from test_quality import fixture_item, obj, pred


def labeled_item(status='accepted', polygon=True):
    jid, iid = fixture_item()
    o = obj(b=[13, 21, 73, 84])
    if polygon:
        o['polygon'] = [[13, 21], [73, 21], [69, 84], [13, 80]]
    db.execute('UPDATE items SET status=?,result=? WHERE id=?', (status, db.dumps(pred(o)), iid))
    return jid, iid


@pytest.mark.parametrize('fmt', ['yolo', 'yolo-seg', 'coco'])
def test_preview_reads_back_the_same_labels_as_export_without_running_models(fmt):
    jid, iid = labeled_item()
    before = db.rows('SELECT * FROM items')
    client = TestClient(app)
    response = client.get(f'/api/items/{iid}/format-preview', params={'format': fmt})
    assert response.status_code == 200
    preview = response.json()
    assert preview['included'] and len(preview['objects']) == 1
    assert preview['objects'][0]['class_id'] == 0
    assert db.rows('SELECT * FROM items') == before
    assert db.one('SELECT count(*) n FROM runs')['n'] == 0
    with zipfile.ZipFile(build_export(jid, fmt, val_ratio=0)) as archive:
        if fmt == 'coco':
            actual = json.loads(archive.read('coco/annotations/instances_train.json'))
            shown = json.loads(preview['annotation'])
            assert shown['annotations'] == actual['annotations']
            assert shown['categories'] == actual['categories']
        else:
            assert preview['annotation'] == archive.read(f'{fmt}/labels/train/{iid}.txt').decode()
    if fmt == 'yolo':
        assert preview['objects'][0]['polygon'] == []
        assert preview['objects'][0]['bbox'] == pytest.approx([13, 21, 73, 84], abs=0.0001)
    else:
        assert len(preview['objects'][0]['polygon']) == 4


def test_preview_exclusions_match_uncertain_and_incomplete_segmentation_policy():
    jid, iid = labeled_item(status='uncertain', polygon=False)
    client = TestClient(app)
    assert not client.get(f'/api/items/{iid}/format-preview').json()['included']
    assert client.get(f'/api/items/{iid}/format-preview?include_uncertain=true').json()['included']
    result = client.get(f'/api/items/{iid}/format-preview?format=yolo-seg&include_uncertain=true').json()
    assert not result['included'] and not result['format_eligible']
    assert result['objects'] == [] and result['annotation'] == '' and '整张图片' in result['format_reason']
    coco = client.get(f'/api/items/{iid}/format-preview?format=coco&include_uncertain=true').json()
    assert coco['included'] and coco['objects'][0]['polygon'] == []
    assert client.get(f'/api/items/{iid}/format-preview?format=pose').status_code == 400
    db.set_job(jid, 'running')
    assert not client.get(f'/api/items/{iid}/format-preview?include_uncertain=true').json()['included']


def test_empty_approved_negative_has_empty_yolo_label_and_is_exportable():
    jid, iid = labeled_item(status='approved')
    db.execute('UPDATE items SET result=? WHERE id=?', (db.dumps(pred()), iid))
    client = TestClient(app)
    for fmt in ['yolo', 'yolo-seg', 'coco']:
        result = client.get(f'/api/items/{iid}/format-preview?format={fmt}').json()
        assert result['included'] and result['objects'] == []
    with zipfile.ZipFile(build_export(jid, 'yolo-seg', val_ratio=0)) as archive:
        assert archive.read(f'yolo-seg/labels/train/{iid}.txt') == b''


def test_trash_and_restore_preserve_pixels_results_and_evidence():
    jid, iid = labeled_item()
    db.save_run(iid, 0, 'generator', 'A', {'prediction': pred(obj()), 'raw': 'independent evidence'})
    image = DATA / db.item(iid)['path']
    pixels = image.read_bytes()
    original = db.item(iid)
    client = TestClient(app)
    assert client.delete(f'/api/jobs/{jid}').status_code == 200
    assert db.job(jid) is None
    assert client.get('/api/jobs').json() == []
    assert client.get('/api/trash').json()[0]['id'] == jid
    for path in [f'/api/jobs/{jid}', f'/api/jobs/{jid}/items', f'/api/items/{iid}',
                 f'/api/items/{iid}/image', f'/api/items/{iid}/format-preview']:
        assert client.get(path).status_code == 404
    assert client.post(f'/api/jobs/{jid}/action', json={'action': 'resume'}).status_code == 404
    with pytest.raises(ValueError, match='任务不存在'):
        build_export(jid)
    assert client.delete(f'/api/jobs/{jid}').status_code == 404
    assert client.post(f'/api/trash/{jid}/restore').status_code == 200
    assert client.get('/api/trash').json() == []
    assert client.get('/api/jobs').json()[0]['status'] == 'paused'
    assert client.post(f'/api/trash/{jid}/restore').status_code == 404
    assert image.read_bytes() == pixels and db.item(iid) == original
    assert db.one('SELECT raw FROM runs WHERE item_id=?', (iid,))['raw'] == 'independent evidence'
    assert [r['kind'] for r in db.rows('SELECT kind FROM audit WHERE job_id=? ORDER BY id', (jid,))] == ['trash', 'restore']


@pytest.mark.parametrize('status', ['importing', 'queued', 'running', 'pausing'])
def test_busy_jobs_cannot_be_deleted(status):
    jid, iid = labeled_item()
    db.set_job(jid, status)
    client = TestClient(app)
    assert client.delete(f'/api/jobs/{jid}').status_code == 409
    assert db.job(jid)['status'] == status and db.item(iid)
    assert client.get('/api/trash').json() == []


def test_old_database_migration_preserves_jobs_and_is_repeatable(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA', tmp_path)
    with sqlite3.connect(tmp_path/'labeler.sqlite3') as con:
        con.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY,name TEXT NOT NULL,status TEXT NOT NULL,config TEXT NOT NULL,created REAL NOT NULL,updated REAL NOT NULL,error TEXT NOT NULL DEFAULT '')")
        con.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)', ('legacy', '旧任务', 'completed', '{}', 1, 1, ''))
    db.init()
    assert db.job('legacy')['name'] == '旧任务' and db.job('legacy')['deleted_at'] == 0
    db.execute('UPDATE jobs SET deleted_at=123 WHERE id=?', ('legacy',))
    db.init()
    assert db.job('legacy') is None and db.job('legacy', include_deleted=True)['deleted_at'] == 123
