"""One-shot acceptance runner for the saved evaluation jobs, no scheduling."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import DATA,ROOT,load_config,readiness
from app.inference import http_json
for n in range(180):
 if readiness(load_config())['ready']:break
 time.sleep(5)
else:raise RuntimeError('Model download did not complete within 15 minutes')
ids=json.loads((DATA/'validation-jobs.json').read_text())
print('weights ready; starting independent reviewer stage',flush=True)
subprocess.run([sys.executable,str(ROOT/'scripts/run_stage.py'),ids[0],'--role','reviewer'],check=True)
for jid in ids:
 print('resume',jid,http_json(f'http://127.0.0.1:7860/api/jobs/{jid}/action',{'action':'resume'})['status'],flush=True)
