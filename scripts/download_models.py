#!/usr/bin/env python3
"""Direct-only resumable ranged downloads; verify byte ranges and SHA-256."""
import concurrent.futures,hashlib,json,os,time,urllib.request,urllib.error,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=[('InternVL3_5-14B.mmproj-f16.gguf',705336928,'b576147e465a94486757253683964229ddb786056af006b32660341253b927c5'),('InternVL3_5-14B.Q8_0.gguf',15698531424,'2eab28cf8bd7ee584c53ceab5d7af0cedbc7297858044bd0b8f15b6fac3f90a0')]
REV='2fd50d9ef8feb79d3c327d8465e82a3df572044c'
BASE=f'https://hf-mirror.com/mradermacher/InternVL3_5-14B-GGUF/resolve/{REV}/'
CHUNK=32*1024*1024

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def download(name,size,digest):
 dest=ROOT/'models'/name;dest.parent.mkdir(exist_ok=True)
 if dest.exists() and dest.stat().st_size==size and sha(dest)==digest:
  print(name,'already verified',flush=True);return
 parts=dest.with_suffix('.parts');parts.mkdir(exist_ok=True)
 start_time=time.time();total=(size+CHUNK-1)//CHUNK
 sourcefile=ROOT/'models'/'download-sources.json'
 sources=json.loads(sourcefile.read_text()) if sourcefile.exists() else {}
 resolved=sources.get(name)
 if not resolved:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  for tries in range(8):
   try:
    with opener.open(urllib.request.Request(BASE+name+'?download=true',method='HEAD',headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as response:resolved=response.url
    break
   except Exception:
    if tries==7:raise
    time.sleep(15+tries*10)
  sources[name]=resolved;sourcefile.write_text(json.dumps(sources))
 url_state={'url':resolved};url_lock=threading.Lock()
 def get(i):
  lo=i*CHUNK;hi=min(size,lo+CHUNK)-1;p=parts/str(i)
  if p.exists() and p.stat().st_size==hi-lo+1:return
  for attempt in range(8):
   try:
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=urllib.request.Request(url_state['url'],headers={'Range':f'bytes={lo}-{hi}','User-Agent':'Mozilla/5.0'})
    with op.open(req,timeout=90) as r:
     cr=r.headers.get('Content-Range','')
     if r.status!=206 or not cr.startswith(f'bytes {lo}-{hi}/'):raise RuntimeError(f'Invalid range {r.status} {cr}')
     tmp=p.with_suffix('.tmp')
     with tmp.open('wb') as f:
      remaining=hi-lo+1
      while remaining:
       b=r.read(min(1024*1024,remaining))
       if not b:raise RuntimeError('short response')
       f.write(b);remaining-=len(b)
     tmp.replace(p)
    if i%10==0:print(name,f'part {i+1}/{total}',round(time.time()-start_time),'seconds',flush=True)
    return
   except Exception as e:
    print(name,'retry',i,attempt,type(e).__name__,getattr(e,'code',''),flush=True)
    if isinstance(e,urllib.error.HTTPError) and e.code in (401,403):
     with url_lock:
      try:
       opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
       with opener.open(urllib.request.Request(BASE+name+'?download=true',method='HEAD'),timeout=60) as response:url_state['url']=response.url
      except Exception:pass
    time.sleep(min(30,2**attempt))
  raise RuntimeError(f'Failed chunk {i}')
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(get,range(total)))
 tmp=dest.with_suffix('.downloading')
 with tmp.open('wb') as out:
  for i in range(total):
   with (parts/str(i)).open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):out.write(b)
 if sha(tmp)!=digest:raise RuntimeError('SHA256 mismatch; retained chunks for diagnosis')
 tmp.replace(dest)
 for p in parts.iterdir():p.unlink()
 parts.rmdir()
 print(name,'VERIFIED',digest,'seconds',round(time.time()-start_time),flush=True)
if __name__=='__main__':
 for spec in FILES:download(*spec)
 (ROOT/'models'/'download-complete.json').write_text(json.dumps({'revision':REV,'files':FILES,'proxy':False}))
