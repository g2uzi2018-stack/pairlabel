from fastapi.testclient import TestClient
from app import db
from app.main import app
from app.progress import summarize
from test_quality import fixture_item, obj, pred


def item(i, attempt=0, status='pending', result=None):
    return {'id': i, 'attempt': attempt, 'status': status, 'result_attempt': result}


def run(i, role='generator', attempt=0, status='ok'):
    return {'item_id': i, 'role': role, 'attempt': attempt, 'status': status}


def test_partial_generator_progress_is_visible_before_any_final_labels():
    p = summarize([item('a'), item('b')], [run('a')])
    assert p['round'] == 1 and p['total'] == 2
    assert p['generator']['done'] == 1
    assert p['reviewer']['done'] == p['reviewed']['done'] == 0


def test_failed_model_calls_are_not_presented_as_successful_identification():
    p = summarize([item('a'), item('b')], [run('a'), run('b', status='error')])
    assert p['generator'] == {'done': 1, 'failed': 1, 'total': 2}


def test_round_does_not_change_until_remaining_first_round_images_are_reviewed():
    p = summarize([item('a', attempt=1, result=0), item('b')], [run('a'), run('b')])
    assert p['round'] == 1 and p['total'] == 2
    assert p['reviewed']['done'] == 1 and p['generator']['done'] == 2


def test_retry_round_counts_only_its_images_and_does_not_add_previous_calls():
    items = [item('a', attempt=1, result=0), item('b', status='accepted', result=0)]
    runs = [run('a'), run('b'), run('a', 'reviewer'), run('b', 'reviewer'), run('a', attempt=1)]
    p = summarize(items, runs)
    assert p['round'] == 2 and p['total'] == 1
    assert p['generator']['done'] == 1 and p['reviewer']['done'] == 0
    assert p['reviewed']['done'] == 0 and p['retry_runs'] == 1
    items[0].update(status='uncertain', result_attempt=1)
    assert summarize(items, runs)['reviewed']['done'] == 1


def test_empty_job_has_no_made_up_work():
    p = summarize([], [])
    assert p['total'] == p['generator']['done'] == p['reviewed']['done'] == 0


def test_job_api_exposes_saved_stage_results_without_mutating_or_running_models():
    jid, iid = fixture_item()
    db.save_run(iid, 0, 'generator', 'A', {'prediction': pred(obj())})
    before = db.rows('SELECT * FROM runs')
    client = TestClient(app)
    j = client.get('/api/jobs/' + jid).json()
    assert j['processed'] == 0 and j['progress']['generator']['done'] == 1
    assert client.get('/api/jobs').json()[0]['progress'] == j['progress']
    assert db.rows('SELECT * FROM runs') == before


def test_status_associates_runtime_with_active_job_not_selected_historical_job():
    jid, _ = fixture_item()
    client = TestClient(app)
    assert client.get('/api/status').json()['active_job_id'] is None
    db.set_job(jid, 'running')
    assert client.get('/api/status').json()['active_job_id'] == jid
    db.set_job(jid, 'paused')
    assert client.get('/api/status').json()['active_job_id'] is None
