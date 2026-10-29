import contextlib,json,sqlite3,time,uuid
from .config import DATA

def dumps(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
@contextlib.contextmanager
def connection():
 DATA.mkdir(parents=True,exist_ok=True)
 con=sqlite3.connect(DATA/'labeler.sqlite3',timeout=30)
 con.row_factory=sqlite3.Row
 con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA foreign_keys=ON');con.execute('PRAGMA busy_timeout=30000')
 try:
  yield con;con.commit()
 except Exception:
  con.rollback();raise
 finally:con.close()

def init():
 with connection() as c:
  c.executescript('''
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,name TEXT NOT NULL,status TEXT NOT NULL,config TEXT NOT NULL,created REAL NOT NULL,updated REAL NOT NULL,error TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS items(id TEXT PRIMARY KEY,job_id TEXT NOT NULL REFERENCES jobs(id),name TEXT NOT NULL,hash TEXT NOT NULL,path TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,status TEXT NOT NULL DEFAULT 'pending',attempt INTEGER NOT NULL DEFAULT 0,result TEXT,issues TEXT NOT NULL DEFAULT '[]',error TEXT NOT NULL DEFAULT '',UNIQUE(job_id,hash));
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT,item_id TEXT NOT NULL REFERENCES items(id),attempt INTEGER NOT NULL,role TEXT NOT NULL,status TEXT NOT NULL,model TEXT NOT NULL,prediction TEXT,raw TEXT NOT NULL DEFAULT '',metrics TEXT NOT NULL DEFAULT '{}',error TEXT NOT NULL DEFAULT '',created REAL NOT NULL,UNIQUE(item_id,attempt,role));
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL,item_id TEXT,kind TEXT NOT NULL,detail TEXT NOT NULL,created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS items_job ON items(job_id,status);
''')
  if 'deleted_at' not in {r['name'] for r in c.execute('PRAGMA table_info(jobs)')}:
   c.execute('ALTER TABLE jobs ADD COLUMN deleted_at REAL NOT NULL DEFAULT 0')
  # An interrupted model call is retried; completed runs remain durable.
  c.execute("UPDATE jobs SET status='paused',updated=? WHERE status='pausing'",(time.time(),))
  c.execute("UPDATE jobs SET status='queued',updated=? WHERE status='running'",(time.time(),))
  c.execute("UPDATE jobs SET status='paused',error='导入曾中断；已导入图片可继续，未导入部分请另建任务',updated=? WHERE status='importing'",(time.time(),))

def rows(sql,args=()):
 with connection() as c:return [dict(r) for r in c.execute(sql,args).fetchall()]

def one(sql,args=()):
 r=rows(sql,args);return r[0] if r else None

def execute(sql,args=()):
 with connection() as c:return c.execute(sql,args).lastrowid

def audit(jid,iid,kind,detail):execute('INSERT INTO audit(job_id,item_id,kind,detail,created) VALUES(?,?,?,?,?)',(jid,iid,kind,dumps(detail),time.time()))

def job(jid,include_deleted=False):
 j=one('SELECT * FROM jobs WHERE id=?'+('' if include_deleted else ' AND deleted_at=0'),(jid,))
 if not j:return None
 j['config']=json.loads(j['config']);j['counts']={r['status']:r['n'] for r in rows('SELECT status,count(*) n FROM items WHERE job_id=? GROUP BY status',(jid,))};j['total']=sum(j['counts'].values())
 j['processed']=sum(j['counts'].get(s,0) for s in ['accepted','uncertain','approved','rejected','error'])
 j['runs']=one("SELECT count(*) n FROM runs JOIN items ON runs.item_id=items.id WHERE items.job_id=? AND runs.status='ok'",(jid,))['n']
 from .progress import summarize
 # Only scalar metadata is read; polling never loads image pixels or model outputs.
 items=rows("SELECT id,attempt,status,json_extract(result,'$.attempt') result_attempt FROM items WHERE job_id=?",(jid,))
 runs=rows('SELECT r.item_id,r.attempt,r.role,r.status FROM runs r JOIN items i ON i.id=r.item_id WHERE i.job_id=?',(jid,))
 j['progress']=summarize(items,runs)
 from .review import FAILED_SQL
 failed=one(f'SELECT count(*) n FROM items WHERE job_id=? AND {FAILED_SQL}',(jid,))['n']
 j['review_counts']={'failed':failed,'uncertain':j['counts'].get('uncertain',0)+j['counts'].get('error',0)-failed}
 return j

def item(iid):
 i=one('SELECT * FROM items WHERE id=?',(iid,))
 if i:
  for k in ['result','issues']:i[k]=json.loads(i[k]) if i[k] else None
  from .review import recognition_failed,review_revision
  i['recognition_failed']=recognition_failed(i)
  i['review_revision']=review_revision(i)
 return i

def set_job(jid,status,error=''):execute('UPDATE jobs SET status=?,error=?,updated=? WHERE id=? AND deleted_at=0',(status,error,time.time(),jid))

def new_job(name,config):
 jid=uuid.uuid4().hex;now=time.time()
 execute('INSERT INTO jobs(id,name,status,config,created,updated) VALUES(?,?,?,?,?,?)',(jid,name,'importing',dumps(config),now,now));return jid

def save_run(iid,attempt,role,model,out=None,error=''):
 out=out or {};execute('INSERT INTO runs(item_id,attempt,role,status,model,prediction,raw,metrics,error,created) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(item_id,attempt,role) DO UPDATE SET status=excluded.status,prediction=excluded.prediction,raw=excluded.raw,metrics=excluded.metrics,error=excluded.error,created=excluded.created',
 (iid,attempt,role,'error' if error else 'ok',model,dumps(out.get('prediction')),out.get('raw',''),dumps(out.get('metrics',{})),error,time.time()))
