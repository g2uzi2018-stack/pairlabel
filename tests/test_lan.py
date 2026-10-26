import json
import os
import subprocess
import sys
from fastapi.testclient import TestClient
from app.main import app


def test_default_listener_remains_loopback():
    response = TestClient(app).get('/api/health')
    assert response.status_code == 200
    assert response.json()['local_only']
    assert TestClient(app, base_url='http://unrelated.example').get('/api/health').status_code == 400


def test_configured_wired_host_is_accepted_and_cross_origin_writes_still_rejected():
    # A fresh process tests the real application middleware, with its startup environment.
    code = '''
import json
from fastapi.testclient import TestClient
from app.main import app
c=TestClient(app,base_url="http://192.0.2.10:7860")
health=c.get("/api/health")
same=c.post("/api/prompt-preview",json={},headers={"origin":"http://192.0.2.10:7860"})
cross=c.post("/api/prompt-preview",json={},headers={"origin":"http://unrelated.example"})
untrusted=c.get("/api/health",headers={"host":"192.0.2.11:7860"})
print(json.dumps({"health":health.json(),"health_status":health.status_code,"same":same.status_code,"cross":cross.status_code,"untrusted":untrusted.status_code}))
'''
    env = {**os.environ, 'LABELER_HOST': '192.0.2.10'}
    out = subprocess.run([sys.executable, '-c', code], env=env, text=True, capture_output=True, check=True)
    result = json.loads(out.stdout)
    assert result['health_status'] == 200
    assert result['health'] == {'status': 'ok', 'local_only': False, 'listen_host': '192.0.2.10'}
    assert result['same'] == 200
    assert result['cross'] == 403 and result['untrusted'] == 400
