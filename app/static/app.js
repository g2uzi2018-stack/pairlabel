'use strict';
const $=id=>document.getElementById(id);const COLORS=['#e45b4b','#3988c4','#20a47d','#ae7bdb','#d58c29','#21a2b5','#d85d97'];
const LABELS={pending:'待处理',accepted:'自动通过',uncertain:'疑难',approved:'人工确认',rejected:'已排除',error:'识别失败',recognition_failed:'识别失败',queued:'等待中',running:'处理中',paused:'已暂停',pausing:'正在暂停',completed:'已完成',failed:'任务失败',importing:'导入中'};
let currentJob=null,currentItem=null,currentLayer='final',filter='all',page=0,pageTotal=0,config=null,jobsCache=[],uploadFiles=[],reviewAction='approve',pollBusy=false,renderSeq=0;let viewport={zoom:1,cx:0,cy:0};
let runtimeStatus=null,labelDraft=null,reviewBusy=false,visibleItemIds=[],imageNavBusy=false;const labelDrafts=new Map();
let currentFormat='yolo',formatPreview=null,previewSeq=0,itemLoadSeq=0,jobLoadSeq=0;
let annotationPresets={},formProfile='general',taskDrafts={},annotationLoadSeq=0,taskClassCatalog=[];
function profileOf(c){return c.annotation_profile||((c.classes||[]).length&&(c.classes||[]).every(x=>x.piece&&x.color)?'xiangqi':'general')}
function profileLabel(c){return profileOf(c)==='xiangqi'?'象棋增强':'通用物件'}
function serializeClasses(classes){return classes.map(c=>c.name===c.label?c.label:`${c.name} | ${c.label}`).join('\n')}
function parseClasses(text,profile){
 const lines=text.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);
 if(!lines.length||lines.length>200)throw Error('请填写 1–200 个类别，每行一类');
 const names=new Set(),known=[...taskClassCatalog,...(annotationPresets.xiangqi?.classes||[])].filter(c=>c.piece&&['red','black'].includes(c.color));
 return lines.map((line,id)=>{
  const parts=line.split('|').map(x=>x.trim());if(parts.length>2||parts.some(x=>!x))throw Error(`第 ${id+1} 行格式有误，请写“类别”或“name | 显示名称”`);
  let name=parts[0],label=parts[1]||name,metadata={};
  if(profile==='xiangqi'){const match=known.find(c=>c.name===name)||known.find(c=>c.label===label);if(match){metadata=match;if(parts.length===1)name=match.name}}
  if(names.has(name))throw Error(`类别名称重复：${name}`);names.add(name);return {...metadata,id,name,label};
 });
}
function showProfileHelp(){$('taskProfileHelp').textContent=formProfile==='xiangqi'?'加入读字、红黑与棋子数量检查；可从象棋预设选取类别。':'类别与补充要求决定标注目标，可自由增删物件类别。';$('taskPromptText').textContent='配置已更新，点击更新预览。'}
function fillTaskDefinition(c){taskClassCatalog=c.classes.map(x=>({...x}));formProfile=profileOf(c);taskDrafts={};$('taskProfile').value=formProfile;$('taskClasses').value=serializeClasses(c.classes);$('taskInstructions').value=c.annotation_instructions||'';showProfileHelp()}
function chooseTaskProfile(profile,reset=false){
 taskDrafts[formProfile]={classes:$('taskClasses').value,instructions:$('taskInstructions').value};formProfile=profile;$('taskProfile').value=profile;
 const draft=taskDrafts[profile];$('taskClasses').value=!reset&&draft?draft.classes:serializeClasses(annotationPresets[profile]?.classes||[]);$('taskInstructions').value=draft?.instructions||'';showProfileHelp();
}
function taskOptions(){return {annotation_profile:$('taskProfile').value,classes:parseClasses($('taskClasses').value,$('taskProfile').value),annotation_instructions:$('taskInstructions').value.trim(),segmentation:$('segmentation').checked,max_attempts:$('autoRetry').checked?2:1}}
async function loadAnnotationDefaults(){const seq=++annotationLoadSeq,controls=Array.from($('importForm').elements);controls.forEach(el=>el.disabled=true);try{const[c,presets]=await Promise.all([api('/api/config'),api('/api/presets')]);if(seq!==annotationLoadSeq)return;annotationPresets=presets;fillTaskDefinition(c)}finally{if(seq===annotationLoadSeq)controls.forEach(el=>el.disabled=false)}}
$('taskProfile').onchange=()=>chooseTaskProfile($('taskProfile').value);
$('generalPreset').onclick=()=>chooseTaskProfile('general',true);$('xiangqiPreset').onclick=()=>chooseTaskProfile('xiangqi',true);
$('taskClasses').oninput=()=>{$('taskPromptText').textContent='类别已更新，点击更新预览。'};$('taskInstructions').oninput=()=>{$('taskPromptText').textContent='要求已更新，点击更新预览。'};
$('previewPromptBtn').onclick=async()=>{try{$('previewPromptBtn').disabled=true;const r=await api('/api/prompt-preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(taskOptions())});$('taskPromptText').textContent=r.independence+'\n\n'+r.prompt}catch(e){$('taskPromptText').textContent=e.message}finally{$('previewPromptBtn').disabled=false}};
async function api(path,options={}){const response=await fetch(path,options);if(!response.ok){let detail;try{detail=(await response.json()).detail}catch{detail=`请求失败 (${response.status})`}throw Error(typeof detail==='string'?detail:JSON.stringify(detail))}return response.json()}
function toast(text,error=false){$('toast').textContent=text;$('toast').className=error?'error':'';$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,6000)}
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function pill(status){return node('span',LABELS[status]||status,'status-pill '+status)}
function setJobVisible(show){$('workspace').hidden=!show;$('importPanel').hidden=show}

function stageSummary(j){
 const p=j.progress;if(!p)return `最终 ${j.processed}/${j.total} 张`;
 const a=p.generator,b=p.reviewer;
 if(['completed','importing'].includes(j.status))return `最终 ${j.processed}/${j.total} 张`;
 const label=a.done+a.failed<p.total?`A ${a.done}/${p.total}`:b.done+b.failed<p.total?`B ${b.done}/${p.total}`:`复核 ${p.reviewed.done}/${p.total}`;
 return `第 ${p.round} 轮 · ${label} · 最终 ${j.processed}/${j.total}`;
}
function durationText(seconds){const n=Math.max(0,Math.floor(seconds));return n<60?`${n} 秒`:`${Math.floor(n/60)} 分 ${n%60} 秒`}
function renderProgress(){
 const j=currentJob;if(!j)return;const p=j.progress;
 $('stageProgress').replaceChildren();
 if(p){
  $('roundTitle').textContent=`第 ${p.round} 轮处理进度${p.round>1?' · 疑难重试':''}`;
  for(const [key,label]of[['generator','模型 A · 独立识别'],['reviewer','模型 B · 独立识别'],['reviewed',j.config.segmentation?'轮廓与复核':'结果复核']]){
   const data=p[key],card=node('div',undefined,'stage-card');
   const heading=node('div',undefined,'stage-heading');heading.append(node('span',label),node('strong',`${data.done}/${data.total} 张`));
   const track=node('div',undefined,'stage-track'),bar=node('div');bar.style.width=(data.total?data.done/data.total*100:0)+'%';track.append(bar);card.append(heading,track);
   if(data.failed)card.append(node('small',`${data.failed} 张识别失败，待复核或重试`,'stage-error'));
   $('stageProgress').append(card);
  }
 }
 $('finalProgressLabel').textContent=`最终标注完成 ${j.processed}/${j.total} 张`;
 $('progressBar').style.width=(j.total?j.processed/j.total*100:0)+'%';
 const own=runtimeStatus?.active_job_id===j.id&&['running','pausing'].includes(j.status);
 if(own){
  const r=runtimeStatus.runtime,active=r.progress||{},parts=[r.message];
  if(active.image)parts.push(active.image);
  if(active.attempt)parts.push(`第 ${active.attempt} 轮`);
  if(active.tiles>1)parts.push(`分块 ${active.tile}/${active.tiles}`);
  if(active.started)parts.push(`本次图片处理已用 ${durationText(Date.now()/1000-active.started)}（含加载）`);
  $('phaseText').textContent=parts.join(' · ');
 }else $('phaseText').textContent=({completed:'全部图片处理结束，疑难结果已单独保留。',paused:'任务已暂停，已保存的阶段结果会在继续时复用。',pausing:'等待当前图片结果保存后暂停。',queued:'任务已排队，等待后台处理。',failed:'任务停止，请查看错误信息。',importing:'正在导入图片，请保持页面打开。',running:'后台正在处理，正在同步当前图片。'})[j.status]||'';
}


function itemState(i){return i.recognition_failed?'recognition_failed':i.status}
function rememberLabelDraft(){
 if(!labelDraft)return;
 if(labelDraft.ids.some((cid,n)=>cid!==labelDraft.base[n]))labelDrafts.set(labelDraft.itemId,{itemId:labelDraft.itemId,base:[...labelDraft.base],ids:[...labelDraft.ids]});
 else labelDrafts.delete(labelDraft.itemId);
}
function activateReview(){
 if(!currentItem)return;
 if(labelDraft?.itemId!==currentItem.id){
  rememberLabelDraft();const ids=(currentItem.result?.objects||[]).map(o=>o.class_id),cached=labelDrafts.get(currentItem.id);
  if(cached&&cached.base.length===ids.length&&cached.base.every((cid,n)=>cid===ids[n]))labelDraft={itemId:cached.itemId,base:[...cached.base],ids:[...cached.ids]};
  else{labelDrafts.delete(currentItem.id);labelDraft={itemId:currentItem.id,base:[...ids],ids:[...ids]}}
 }
}
function renderImageNavigation(){
 const index=currentItem?visibleItemIds.indexOf(currentItem.id):-1,busy=imageNavBusy||reviewBusy||!currentItem;
 $('prevImageBtn').disabled=busy||(index<=0&&page===0);
 $('nextImageBtn').disabled=busy||(!visibleItemIds.length)||((index>=visibleItemIds.length-1)&&(page+1)*30>=pageTotal);
}
async function navigateImage(direction){
 if(imageNavBusy||reviewBusy||!currentItem||!currentJob||![-1,1].includes(direction))return;
 const jid=currentJob.id,selectedFilter=filter,index=visibleItemIds.indexOf(currentItem.id);
 imageNavBusy=true;renderImageNavigation();
 try{
  const target=index+direction;
  if(index<0&&visibleItemIds.length){await selectItem(visibleItemIds[direction>0?0:visibleItemIds.length-1]);return}
  if(target>=0&&target<visibleItemIds.length){await selectItem(visibleItemIds[target]);return}
  const hasPage=direction<0?page>0:(page+1)*30<pageTotal;
  if(!hasPage){toast(direction<0?'已到当前筛选的第一张':'已到当前筛选的最后一张');return}
  page+=direction;await refreshItems();
  if(currentJob?.id!==jid||filter!==selectedFilter)return;
  const id=visibleItemIds[direction<0?visibleItemIds.length-1:0];if(id)await selectItem(id);
 }catch(e){toast(e.message,true)}finally{imageNavBusy=false;renderImageNavigation()}
}
$('prevImageBtn').onclick=()=>navigateImage(-1);$('nextImageBtn').onclick=()=>navigateImage(1);
document.addEventListener('keydown',e=>{
 const direction={ArrowLeft:-1,ArrowUp:-1,ArrowRight:1,ArrowDown:1}[e.key];
 if(!direction||e.defaultPrevented||e.isComposing||e.altKey||e.ctrlKey||e.metaKey||e.shiftKey||!currentJob||!currentItem||$('workspace').hidden)return;
 const target=e.target?.nodeType===1?e.target:document.activeElement;
 if(target?.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"]),[role="combobox"],[role="textbox"]')||document.querySelector('dialog[open]'))return;
 e.preventDefault();if(!e.repeat)navigateImage(direction);
});
function chooseLayer(layer){currentLayer=layer;if(layer==='review')activateReview();document.querySelectorAll('#layerTabs button').forEach(b=>b.classList.toggle('active',b.dataset.layer===layer));renderItem();refreshFormatPreview()}
function focusObject(o){const b=o.bbox;viewport={zoom:Math.min(8,Math.max(2,Math.min(currentItem.width/((b[2]-b[0])*3),currentItem.height/((b[3]-b[1])*3)))),cx:(b[0]+b[2])/2,cy:(b[1]+b[3])/2};applyViewport()}
function renderObjectList(objects,classes){
 const list=$('objectList');list.replaceChildren();
 for(const [n,o]of objects.entries()){
  const row=node('div',undefined,'object-line'+(currentLayer==='review'?' review-object':''));
  if(currentLayer==='review'){
   const zoom=node('button',undefined,'target-crop');zoom.type='button';zoom.title=`放大目标 ${n+1}`;zoom.setAttribute('aria-label',zoom.title);
   const b=o.bbox,pad=Math.max(b[2]-b[0],b[3]-b[1])*.18,w=b[2]-b[0]+pad*2,h=b[3]-b[1]+pad*2;
   const svg=svgEl('svg',{viewBox:`${b[0]-pad} ${b[1]-pad} ${w} ${h}`,'aria-hidden':'true'});svg.append(svgEl('image',{href:`/api/items/${currentItem.id}/view`,width:currentItem.width,height:currentItem.height}));zoom.append(svg);zoom.onclick=()=>focusObject(o);
   const copy=node('div',undefined,'target-category');copy.append(node('label',`目标 ${n+1}`));
   const select=node('select');select.dataset.reviewIndex=n;select.setAttribute('aria-label',`目标 ${n+1} 类别`);
   for(const c of classes){const option=node('option',c.label);option.value=c.id;select.append(option)}
   select.value=o.class_id;select.disabled=reviewBusy||['running','queued','pausing','importing'].includes(currentJob.status)||currentItem.status==='pending';
   select.onchange=()=>{labelDraft.ids[n]=Number(select.value);rememberLabelDraft();renderItem();document.querySelector(`select[data-review-index="${n}"]`)?.focus({preventScroll:true})};
   copy.append(select);row.append(zoom,copy,node('small',o.polygon?.length?'有轮廓':'检测框'));
  }else{
   const swatch=node('span',undefined,'color-swatch');swatch.style.background=classes[o.class_id]?.color==='red'?'#df5a4e':classes[o.class_id]?.color==='black'?'#456277':COLORS[o.class_id%COLORS.length];
   row.tabIndex=0;row.setAttribute('role','button');row.onclick=()=>focusObject(o);row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();row.click()}};
   row.append(swatch,node('span',`${n+1}. ${classes[o.class_id]?.label??o.class_id}`),node('small',o.polygon?.length?'有轮廓':'检测框'));
  }
  list.append(row);
 }
 if(!objects.length)list.append(node('p',currentLayer==='review'?'没有可修改的目标，可重试识别或使用高级编辑补充。':'当前图层暂无目标。','hint'));
 const changed=labelDraft?.itemId===currentItem.id?labelDraft.ids.filter((cid,n)=>cid!==labelDraft.base[n]).length:0;
 $('labelDraftNote').textContent=currentLayer==='review'?(changed?`已修改 ${changed} 个类别，尚未保存。`:'核对目标类别后，点击“确认本图”即可保存。'):'';
 $('labelDraftNote').hidden=currentLayer!=='review';
 $('reviewHelpInline').hidden=currentLayer!=='review';
}
function renderFilterCounts(){
 const j=currentJob;if(!j)return;const counts={all:j.total,accepted:j.counts.accepted||0,uncertain:j.review_counts?.uncertain??j.counts.uncertain??0,failed:j.review_counts?.failed??j.counts.error??0,approved:j.counts.approved||0,pending:j.counts.pending||0};
 const labels={all:'全部',accepted:'自动通过',uncertain:'疑难',failed:'识别失败',approved:'人工确认',pending:'待处理'};
 for(const b of document.querySelectorAll('#filters button[data-filter]'))b.textContent=`${labels[b.dataset.filter]} ${counts[b.dataset.filter]}`;
 $('retryFailedBtn').hidden=filter!=='failed';$('retryFailedBtn').disabled=reviewBusy||['queued','running','pausing','importing'].includes(j.status)||!counts.failed;
}
async function confirmLabels(next=false){
 if(!currentItem||reviewBusy)return;activateReview();
 const iid=currentItem.id,jid=currentJob.id,index=visibleItemIds.indexOf(iid);
 const ids=[...labelDraft.ids],expected=[...labelDraft.base];
 reviewBusy=true;renderItem();
 try{
  const saved=await api(`/api/items/${iid}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'labels',class_ids:ids,expected_class_ids:expected})});
  if(currentJob?.id!==jid||currentItem?.id!==iid)return;
  labelDrafts.delete(iid);currentItem=saved;labelDraft=null;activateReview();await refreshJobs();await refreshItems();
  if(next&&visibleItemIds.length){
   const at=visibleItemIds.indexOf(iid),nextIndex=at>=0?at+1:Math.max(0,index);
   if(nextIndex<visibleItemIds.length)await selectItem(visibleItemIds[nextIndex]);
   else if((page+1)*30<pageTotal){page++;await refreshItems(true)}
   else toast('当前筛选已核对到最后一张');
  }
  toast('已人工确认，类别修改已保存');
 }catch(e){toast(e.message,true)}finally{reviewBusy=false;if(currentItem){renderItem();await refreshFormatPreview()}}
}
$('retryFailedBtn').onclick=async()=>{
 if(!currentJob)return;const jid=currentJob.id;$('retryFailedBtn').disabled=true;
 try{currentJob=await api(`/api/jobs/${jid}/action`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'retry_failed'})});renderJob();await refreshItems();toast('仅识别失败的图片已加入重试队列')}catch(e){toast(e.message,true)}finally{renderFilterCounts()}
};

async function refreshJobs(){jobsCache=await api('/api/jobs');$('jobCount').textContent=jobsCache.length;const list=$('jobList');list.replaceChildren();for(const j of jobsCache){const btn=node('button',undefined,'job-entry'+(currentJob?.id===j.id?' active':''));btn.append(node('strong',j.name),node('small',`${LABELS[j.status]||j.status} · ${stageSummary(j)}`));btn.onclick=()=>selectJob(j.id);list.append(btn)}if(currentJob){const latest=jobsCache.find(j=>j.id===currentJob.id);if(latest){const changed=currentJob.status!==latest.status;currentJob=latest;renderJob();if(changed&&currentItem)await refreshFormatPreview()}}}
async function selectJob(id){const seq=++jobLoadSeq;currentJob=null;currentItem=null;formatPreview=null;++renderSeq;++itemLoadSeq;++previewSeq;$('selection').hidden=true;$('noSelection').hidden=false;$('imageList').replaceChildren();$('exportBtn').disabled=true;try{const selected=await api('/api/jobs/'+id);if(seq!==jobLoadSeq)return;currentJob=selected;currentItem=null;page=0;filter='all';currentLayer='final';formatPreview=null;++previewSeq;document.querySelectorAll('#layerTabs button').forEach(b=>b.classList.toggle('active',b.dataset.layer==='final'));setJobVisible(true);document.querySelectorAll('#filters button').forEach(b=>b.classList.toggle('active',b.dataset.filter===filter));renderJob();await refreshJobs();if(seq!==jobLoadSeq)return;await refreshItems(true)}catch(e){if(seq===jobLoadSeq)toast(e.message,true)}}
function renderJob(){const j=currentJob;if(!j)return;$('pageTitle').textContent=j.name;$('pageDesc').textContent=`${j.total} 张图片 · ${new Date(j.created*1000).toLocaleString('zh-CN')} 创建 · ${profileLabel(j.config)} · ${j.config.classes.length} 类`;$('jobClassesSummary').textContent=j.config.classes.map(c=>`${c.id}: ${c.label}`).join('、');$('jobInstructionsSummary').textContent=j.config.annotation_instructions||'未填写补充要求；按本任务类别清单标注可见目标。';$('totalCount').textContent=j.total;$('doneCount').textContent=j.processed;$('acceptedCount').textContent=(j.counts.accepted||0)+(j.counts.approved||0);$('uncertainCount').textContent=(j.counts.uncertain||0)+(j.counts.error||0);$('progressText').textContent=`${j.runs} 次模型输出已保存${j.progress?.retry_runs?' · 重试 '+j.progress.retry_runs+' 次':''}`;$('jobStatus').textContent=LABELS[j.status]||j.status;$('jobStatus').className='status-pill '+j.status;$('progressBar').style.width=(j.total?j.processed/j.total*100:0)+'%';$('pauseBtn').hidden=!['queued','running'].includes(j.status);$('resumeBtn').hidden=!['paused','failed'].includes(j.status);$('retryBtn').disabled=['queued','running','pausing','importing'].includes(j.status)||!((j.counts.uncertain||0)+(j.counts.error||0));$('exportBtn').disabled=['queued','running','pausing','importing'].includes(j.status)||!j.processed;$('deleteJobBtn').disabled=['queued','running','pausing','importing'].includes(j.status);$('deleteJobBtn').title=$('deleteJobBtn').disabled?'先暂停任务，等待保存完成后再删除':'移入回收站，可恢复';$('jobError').hidden=!j.error;$('jobError').textContent=j.error;renderProgress();renderFilterCounts()}
async function refreshStatus(){const s=await api('/api/status');runtimeStatus=s;const m=s.memory_kb;$('memory').textContent=`可用 ${(m.MemAvailable/1024/1024).toFixed(1)} / ${(m.MemTotal/1024/1024).toFixed(0)} GB`;$('modelState').textContent=s.models.ready?`${s.runtime.message} · A: ${s.models.models[0].name} · B: ${s.models.models[1].name}`:`模型准备中：${s.models.missing.join('、')}`;renderProgress();$('modelDetails').textContent=JSON.stringify({project:s.project,runtime:s.runtime,models:s.models},null,2)+(s.download_log?'\n\n直连下载记录：\n'+s.download_log:'')}
async function refreshItems(autoSelect=false){if(!currentJob)return;const seq=++renderSeq;const data=await api(`/api/jobs/${currentJob.id}/items?status=${filter}&offset=${page*30}&limit=30`);if(seq!==renderSeq)return;pageTotal=data.total;visibleItemIds=data.items.map(i=>i.id);renderFilterCounts();renderImageNavigation();$('resultCount').textContent=`${data.total} 张`;$('pageLabel').textContent=`${page+1}/${Math.max(1,Math.ceil(data.total/30))}`;$('prevPage').disabled=page===0;$('nextPage').disabled=(page+1)*30>=data.total;const list=$('imageList');list.replaceChildren();if(!data.items.length)list.append(node('p','当前筛选下没有图片','no-images'));for(const i of data.items){const b=node('button',undefined,'image-row'+(currentItem?.id===i.id?' active':''));const im=node('img');im.src=`/api/items/${i.id}/thumbnail`;im.alt=i.name;im.loading='lazy';const desc=node('div',undefined,'row-copy');desc.append(node('strong',i.name),node('small',`${i.width} × ${i.height} · ${i.result?.objects?.length??'—'} 个目标`),document.createElement('br'),pill(itemState(i)));b.append(im,desc);b.onclick=()=>selectItem(i.id);list.append(b)}if(autoSelect&&data.items.length)await selectItem(data.items[0].id);else if(currentItem){const updated=data.items.find(i=>i.id===currentItem.id);if(updated&&(updated.status!==currentItem.status||updated.attempt!==currentItem.attempt||(updated.status==='pending'&&['running','pausing'].includes(currentJob.status))))await selectItem(currentItem.id)}else{$('noSelection').hidden=false;$('selection').hidden=true}}
async function selectItem(id){try{const seq=++itemLoadSeq,item=await api('/api/items/'+id);if(seq!==itemLoadSeq||!currentJob||item.job_id!==currentJob.id)return;if(currentItem?.id!==id){rememberLabelDraft();viewport={zoom:1,cx:item.width/2,cy:item.height/2};labelDraft=null;if(['final','review'].includes(currentLayer))currentLayer=['uncertain','error'].includes(item.status)?'review':'final'}currentItem=item;activateReview();document.querySelectorAll('#layerTabs button').forEach(b=>b.classList.toggle('active',b.dataset.layer===currentLayer));formatPreview=null;++previewSeq;$('noSelection').hidden=true;$('selection').hidden=false;document.querySelectorAll('.image-row').forEach(b=>b.classList.toggle('active',b.querySelector('img')?.src.includes(id)));renderItem();await refreshFormatPreview()}catch(e){toast(e.message,true)}}
function displayedObjects(){if(!currentItem)return[];if(currentLayer==='review'){activateReview();return (currentItem.result?.objects||[]).map((o,n)=>({...o,class_id:labelDraft.ids[n]}))}if(currentLayer==='final')return formatPreview?.item_id===currentItem.id&&formatPreview.format===currentFormat?formatPreview.objects:[];const candidates=currentItem.runs.filter(r=>r.role===currentLayer&&r.status==='ok');return candidates.at(-1)?.prediction?.objects||[]}
function svgEl(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n}
function renderItem(){const i=currentItem;if(!i)return;const classes=currentJob.config.classes,objects=displayedObjects();const canvas=$('canvas');canvas.replaceChildren();canvas.setAttribute('viewBox',`0 0 ${i.width} ${i.height}`);canvas.append(svgEl('image',{href:`/api/items/${i.id}/view`,x:0,y:0,width:i.width,height:i.height}));objects.forEach((o,index)=>{const color=classes[o.class_id]?.color==='red'?'#df5a4e':classes[o.class_id]?.color==='black'?'#456277':COLORS[o.class_id%COLORS.length],b=o.bbox;if($('showPolygons').checked&&(currentLayer!=='final'||currentFormat!=='yolo')&&o.polygon?.length)canvas.append(svgEl('polygon',{points:o.polygon.map(p=>p.join(',')).join(' '),fill:color+'33',stroke:color,'stroke-width':Math.max(1,i.width/600)}));if(currentLayer!=='final'||currentFormat!=='yolo-seg')canvas.append(svgEl('rect',{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:'none',stroke:color,'stroke-width':Math.max(1.5,i.width/450)}));if($('showLabels').checked)canvas.append(svgEl('text',{x:b[0]+2,y:Math.max(i.height/40,b[1]-5),fill:color,'font-size':Math.max(13,i.width/55),'font-family':'sans-serif','font-weight':'600'},`${index+1} ${classes[o.class_id]?.label??o.class_id}`))});$('imageName').textContent=i.name;$('imageMeta').textContent=`${i.width} × ${i.height} · 第 ${i.attempt+1} 次尝试`;$('itemStatus').textContent=LABELS[itemState(i)];$('itemStatus').className='status-pill '+itemState(i);$('objectCount').textContent=`${objects.length} 个`;$('issues').replaceChildren();if(i.status==='pending')$('issues').append(node('p',`本轮已保存：${i.runs.filter(r=>r.attempt===i.attempt&&r.status==='ok').map(r=>r.role==='generator'?'模型 A':'模型 B').join('、')||'等待识别'}。最终标注需等待双模型和复核完成，可切换模型图层查看已保存的中间结果。`,'hint'));else if(i.issues?.length){i.issues.forEach(issue=>$('issues').append(node('div',issue.message,'issue')))}else $('issues').append(node('div','两个模型的目标、类别与位置通过一致性检查。','pass-note'));if(i.result?.spatial_note)$('issues').append(node('p',i.result.spatial_note,'hint'));if(i.result?.retry_skipped)$('issues').append(node('p',i.result.retry_skipped,'hint'));if(i.result?.human_review)$('issues').append(node('p','人工复核：'+i.result.human_review.reason,'hint'));renderObjectList(objects,classes);$('cropEvidence').replaceChildren();for(const run of i.runs){for(const pg of (run.metrics?.pages||[])){if(!pg.evidence_file)continue;const a=node('a');a.href=`/api/items/${i.id}/crops/${run.attempt}/${run.role}/${pg.page}`;a.target='_blank';a.rel='noopener';const im=node('img');im.src=a.href+'?thumbnail=true';im.loading='lazy';im.alt=`${run.role} 第${run.attempt+1}次候选输入`;a.append(im,node('small',`${run.role} · 第${run.attempt+1}次 · 拼图 ${pg.page+1}`));$('cropEvidence').append(a)}}$('evidence').textContent=JSON.stringify({runs:i.runs,audit:i.audit},null,2);const editable=!['running','queued','pausing','importing'].includes(currentJob.status)&&i.status!=='pending';$('retryItemBtn').disabled=!editable||!['uncertain','error'].includes(i.status)||(currentJob.counts.pending||0)>0;['approveBtn','approveNextBtn'].forEach(id=>$(id).disabled=reviewBusy||!editable||!(i.result?.objects||[]).length);['rejectBtn','editBtn','advancedEditBtn'].forEach(id=>$(id).disabled=reviewBusy||!editable);renderFormatInfo();applyViewport();renderImageNavigation()}
function applyViewport(){if(!currentItem)return;const w=currentItem.width/viewport.zoom,h=currentItem.height/viewport.zoom;viewport.cx=Math.max(w/2,Math.min(currentItem.width-w/2,viewport.cx));viewport.cy=Math.max(h/2,Math.min(currentItem.height-h/2,viewport.cy));$('canvas').setAttribute('viewBox',`${viewport.cx-w/2} ${viewport.cy-h/2} ${w} ${h}`);$('zoomFit').textContent=viewport.zoom===1?'适应':Math.round(viewport.zoom*100)+'%'}
function zoomBy(factor){if(!currentItem)return;viewport.zoom=Math.max(1,Math.min(8,viewport.zoom*factor));applyViewport()}
$('zoomIn').onclick=()=>zoomBy(1.4);$('zoomOut').onclick=()=>zoomBy(1/1.4);$('zoomFit').onclick=()=>{if(currentItem)viewport={zoom:1,cx:currentItem.width/2,cy:currentItem.height/2};applyViewport()};
$('canvas').addEventListener('wheel',e=>{e.preventDefault();zoomBy(e.deltaY<0?1.15:1/1.15)},{passive:false});
let dragging=null;$('canvas').addEventListener('pointerdown',e=>{if(viewport.zoom<=1)return;dragging={x:e.clientX,y:e.clientY};$('canvas').setPointerCapture(e.pointerId)});$('canvas').addEventListener('pointermove',e=>{if(!dragging)return;const matrix=$('canvas').getScreenCTM();viewport.cx-=(e.clientX-dragging.x)/matrix.a;viewport.cy-=(e.clientY-dragging.y)/matrix.d;dragging={x:e.clientX,y:e.clientY};applyViewport()});for(const event of ['pointerup','pointercancel'])$('canvas').addEventListener(event,()=>dragging=null);
$('newJob').onclick=async()=>{++jobLoadSeq;currentJob=null;currentItem=null;formatPreview=null;++previewSeq;++renderSeq;++itemLoadSeq;setJobVisible(false);$('pageTitle').textContent='把标注交给模型，把质量留在眼前。';$('pageDesc').textContent='独立识别，交叉复核。为下一轮训练准备可追溯的数据。';$('exportBtn').disabled=true;try{await Promise.all([loadAnnotationDefaults(),refreshJobs()])}catch(e){toast(e.message,true)}};
function updateFileSummary(){$('fileSummary').textContent=uploadFiles.length?`已选择 ${uploadFiles.length} 张 · ${(uploadFiles.reduce((s,f)=>s+f.size,0)/1024/1024).toFixed(1)} MB`:''}
$('files').onchange=()=>{uploadFiles=Array.from($('files').files);updateFileSummary()};for(const name of ['dragenter','dragover'])$('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.add('dragover')});for(const name of ['dragleave','drop'])$('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.remove('dragover')});$('dropzone').addEventListener('drop',e=>{uploadFiles=Array.from(e.dataTransfer.files);updateFileSummary()});
$('importForm').onsubmit=async e=>{e.preventDefault();let options;try{options=taskOptions()}catch(error){toast(error.message,true);return}const data=new FormData();data.set('name',$('taskName').value||'图片标注 · '+new Date().toLocaleDateString('zh-CN'));data.set('directory',$('directory').value.trim());data.set('options',JSON.stringify(options));for(const f of uploadFiles)data.append('files',f);$('startBtn').disabled=true;$('importFeedback').textContent='正在导入并保存原图，请保持页面打开…';try{const r=await api('/api/jobs',{method:'POST',body:data});toast(`导入 ${r.imported} 张，去重 ${r.duplicates} 张，失败 ${r.errors.length} 张`);$('importFeedback').textContent=r.errors.length?JSON.stringify(r.errors):'';uploadFiles=[];$('files').value='';updateFileSummary();await selectJob(r.job.id)}catch(error){$('importFeedback').textContent=error.message;toast(error.message,true)}finally{$('startBtn').disabled=false}};
for(const [id,action]of[['pauseBtn','pause'],['resumeBtn','resume'],['retryBtn','retry_uncertain']])$(id).onclick=async()=>{try{currentJob=await api(`/api/jobs/${currentJob.id}/action`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});renderJob();if(currentItem){renderItem();await refreshFormatPreview()}toast(action==='pause'?'已请求暂停：当前模型对这张图的整组识别保存后暂停。':'任务已加入处理队列');await refreshItems()}catch(e){toast(e.message,true)}};
$('filters').onclick=async e=>{const b=e.target.closest('button[data-filter]');if(!b)return;filter=b.dataset.filter;page=0;currentItem=null;document.querySelectorAll('#filters button').forEach(x=>x.classList.toggle('active',x===b));await refreshItems(true)};
$('layerTabs').onclick=e=>{const b=e.target.closest('button[data-layer]');if(!b)return;chooseLayer(b.dataset.layer)};$('showLabels').onchange=renderItem;$('showPolygons').onchange=renderItem;$('prevPage').onclick=()=>{page=Math.max(0,page-1);currentItem=null;refreshItems(true)};$('nextPage').onclick=()=>{page++;currentItem=null;refreshItems(true)};
$('exportBtn').onclick=()=>{$('exportFormat').value=currentFormat;$('exportFeedback').textContent='';$('exportDialog').showModal()};async function requestExport(download){if(!currentJob)return;$('downloadBtn').disabled=true;$('saveExportBtn').disabled=true;$('exportFeedback').textContent='正在 Jetson 上整理原图、标注与复核记录…';try{const r=await api(`/api/jobs/${currentJob.id}/export`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({format:$('exportFormat').value,include_uncertain:$('includeUncertain').checked,val_ratio:Number($('valRatio').value)})});const a=node('a','下载这个压缩包');a.href=r.download_url;a.download='PairLabel-'+currentJob.name+'.zip';a.style.color='#176b5b';$('exportFeedback').replaceChildren(node('span','已保存到：'+r.server_path),document.createElement('br'),a);if(download)a.click()}catch(e){$('exportFeedback').textContent=e.message}finally{$('downloadBtn').disabled=false;$('saveExportBtn').disabled=false}}
$('downloadBtn').onclick=()=>requestExport(true);$('saveExportBtn').onclick=()=>requestExport(false);
function renderClassTable(classes){const table=node('table',undefined,'class-table'),head=node('tr');['ID','导出类别名称','显示名称'].forEach(t=>head.append(node('th',t)));table.append(head);for(const c of classes){const row=node('tr');row.dataset.id=c.id;row.append(node('td',c.id));for(const key of ['name','label']){const td=node('td'),input=node('input');input.value=c[key];input.dataset.key=key;input.setAttribute('aria-label',`${c.id} ${key}`);td.append(input);row.append(td)}table.append(row)}$('classTable').replaceChildren(table)}
function syncSettingsFields(c){$('settingsProfile').value=profileOf(c);$('settingsInstructions').value=c.annotation_instructions||'';renderClassTable(c.classes)}
$('settingsBtn').onclick=async()=>{try{const[c,presets]=await Promise.all([api('/api/config'),api('/api/presets')]);config=c;annotationPresets=presets;$('configJson').value=JSON.stringify(config,null,2);syncSettingsFields(config);$('settingsDialog').showModal()}catch(e){toast(e.message,true)}};
function updateSettings(mutator){try{const c=JSON.parse($('configJson').value);mutator(c);$('configJson').value=JSON.stringify(c,null,2);return c}catch(e){toast('请先修正高级 JSON：'+e.message,true);return null}}
$('settingsProfile').onchange=()=>{const c=updateSettings(c=>{c.annotation_profile=$('settingsProfile').value;if(c.annotation_profile==='xiangqi'&&!c.classes.every(x=>x.piece&&x.color))c.classes=annotationPresets.xiangqi.classes});if(c)renderClassTable(c.classes)};
$('settingsInstructions').oninput=()=>updateSettings(c=>{c.annotation_instructions=$('settingsInstructions').value});
for(const[id,profile]of[['settingsGeneralPreset','general'],['settingsXiangqiPreset','xiangqi']])$(id).onclick=()=>{const c=updateSettings(c=>{c.annotation_profile=profile;c.classes=annotationPresets[profile].classes});if(c)syncSettingsFields(c)};
$('configJson').oninput=()=>{try{const c=JSON.parse($('configJson').value);if(Array.isArray(c.classes)&&c.classes.every(x=>x&&typeof x==='object'))syncSettingsFields(c)}catch{}};
$('classTable').oninput=e=>{const input=e.target.closest('input[data-key]'),row=input?.closest('tr[data-id]');if(!row)return;try{const next=JSON.parse($('configJson').value);if(next.classes[Number(row.dataset.id)])next.classes[Number(row.dataset.id)][input.dataset.key]=input.value;$('configJson').value=JSON.stringify(next,null,2)}catch{toast('请先修正高级 JSON，再编辑类别表格',true)}};
$('saveSettings').onclick=async()=>{try{const next=JSON.parse($('configJson').value);await api('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(next)});$('settingsDialog').close();toast('设置已保存，将用于新建任务');await refreshStatus();if(!currentJob)await loadAnnotationDefaults()}catch(e){toast(e.message,true)}};
function openReview(action){if(!currentItem)return;reviewAction=action;$('reviewTitle').textContent={approve:'人工确认标注',reject:'排除这张图片',edit:'编辑检测框与轮廓'}[action];$('reviewHelp').textContent=action==='edit'?'编辑像素坐标 bbox [左,上,右,下]、class_id 和 polygon 点列。删除或新增数组项可调整目标；修改后须填写复核依据。':'请核对图片与目标清单，并记录你的判断依据。原始独立识别结果和分歧仍会保留。';$('annotationJson').hidden=action!=='edit';$('annotationJson').value=JSON.stringify(currentItem.result?.objects||[],null,2);$('reviewReason').value='';$('reviewDialog').showModal()}
$('retryItemBtn').onclick=async()=>{try{currentJob=await api(`/api/jobs/${currentJob.id}/action`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:currentItem.recognition_failed?'retry_failed':'retry_uncertain',item_id:currentItem.id})});renderJob();renderItem();await refreshFormatPreview();toast('这张图片已单独加入重试队列')}catch(e){toast(e.message,true)}};
$('approveBtn').onclick=()=>confirmLabels();$('approveNextBtn').onclick=()=>confirmLabels(true);$('rejectBtn').onclick=()=>openReview('reject');$('editBtn').onclick=()=>chooseLayer('review');$('advancedEditBtn').onclick=()=>openReview('edit');$('saveReview').onclick=async()=>{try{const reason=$('reviewReason').value.trim();if(!reason)throw Error('请填写复核依据');const body={action:reviewAction,reason};if(reviewAction==='edit')body.objects=JSON.parse($('annotationJson').value);currentItem=await api(`/api/items/${currentItem.id}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('reviewDialog').close();await refreshFormatPreview();renderItem();await refreshJobs();await refreshItems();toast('人工复核已保存，原始记录已保留')}catch(e){toast(e.message,true)}};

const FORMAT_NAMES={yolo:'YOLO 检测框','yolo-seg':'YOLO 分割轮廓',coco:'COCO 框与轮廓'};
function renderFormatInfo(){
 const final=currentLayer==='final';$('previewFormat').disabled=!final;$('previewFormat').value=currentFormat;
 $('showPolygons').disabled=final&&currentFormat==='yolo';$('previewIncludeUncertain').disabled=!final;
 $('annotationSourcePanel').hidden=!final;
 if(!final){$('formatPreviewInfo').classList.remove('excluded');$('formatPreviewNote').textContent=currentLayer==='review'?'人工复核：下拉选择目标类别，确认后保存。当前显示原始框与已有轮廓；三种导出预览请切换“最终标注”。':'当前显示模型的独立原始输出；切回“最终标注”查看三种导出格式。';return}
 if(!formatPreview){$('formatPreviewNote').textContent='正在读取当前格式的标注…';$('annotationSource').textContent='';return}
 const r=formatPreview;
 $('formatPreviewNote').textContent=r.error?r.error:`${FORMAT_NAMES[currentFormat]} · ${r.included?'本图可纳入当前导出':'本图不会纳入当前导出'}。`+(r.reasons||[]).join(' ')+(currentFormat==='coco'&&!r.error?' COCO 显示框与已有轮廓。':'');
 $('formatPreviewInfo').classList.toggle('excluded',!r.included);
 $('annotationSource').textContent=r.annotation||'（没有标签输出）';
}
async function refreshFormatPreview(){
 const seq=++previewSeq;if(!currentItem||currentLayer!=='final')return;
 const id=currentItem.id,fmt=currentFormat,include=$('previewIncludeUncertain').checked;
 formatPreview=null;renderItem();
 try{
  const r=await api(`/api/items/${id}/format-preview?format=${fmt}&include_uncertain=${include}`);
  if(seq!==previewSeq||currentItem?.id!==id||currentLayer!=='final'||currentFormat!==fmt)return;
  formatPreview=r;renderItem();
 }catch(e){if(seq===previewSeq&&currentItem?.id===id){formatPreview={item_id:id,format:fmt,objects:[],error:e.message};renderItem()}}
}
$('previewFormat').onchange=()=>{currentFormat=$('previewFormat').value;$('exportFormat').value=currentFormat;refreshFormatPreview()};
$('exportFormat').onchange=()=>{if($('exportFormat').value!=='all'){currentFormat=$('exportFormat').value;refreshFormatPreview()}};
for(const id of ['previewIncludeUncertain','includeUncertain'])$(id).onchange=()=>{const checked=$(id).checked;$('previewIncludeUncertain').checked=checked;$('includeUncertain').checked=checked;refreshFormatPreview()};
$('deleteJobBtn').onclick=()=>{if(!currentJob)return;$('deleteJobDescription').textContent=`删除“${currentJob.name}”（${currentJob.total} 张图片）？`;$('deleteJobFeedback').textContent='';$('deleteJobDialog').showModal()};
$('confirmDeleteJob').onclick=async()=>{
 if(!currentJob)return;const id=currentJob.id;$('confirmDeleteJob').disabled=true;
 try{
  await api(`/api/jobs/${id}`,{method:'DELETE'});$('deleteJobDialog').close();
  await $('newJob').onclick();if(jobsCache.length)await selectJob(jobsCache[0].id);toast('任务已移入回收站，可随时恢复');
 }catch(e){$('deleteJobFeedback').textContent=e.message}finally{$('confirmDeleteJob').disabled=false}
};
async function refreshTrash(){
 const jobs=await api('/api/trash'),list=$('trashList');list.replaceChildren();
 if(!jobs.length)list.append(node('p','回收站为空','hint'));
 for(const job of jobs){
  const row=node('div',undefined,'trash-row'),copy=node('div');copy.append(node('strong',job.name),node('small',`${job.total} 张 · ${new Date(job.deleted_at*1000).toLocaleString('zh-CN')} 删除`));
  const restore=node('button','恢复','secondary small');restore.onclick=async()=>{restore.disabled=true;try{await api(`/api/trash/${job.id}/restore`,{method:'POST'});await refreshJobs();await refreshTrash();toast('任务已恢复；暂停的任务不会自动开始推理')}catch(e){$('trashFeedback').textContent=e.message}finally{restore.disabled=false}};
  row.append(copy,restore);list.append(row);
 }
}
$('trashBtn').onclick=async()=>{$('trashFeedback').textContent='';$('trashDialog').showModal();try{await refreshTrash()}catch(e){$('trashFeedback').textContent=e.message}};

async function poll(){if(pollBusy||imageNavBusy||document.hidden)return;pollBusy=true;try{await Promise.all([refreshStatus(),refreshJobs()]);if(currentJob&&!imageNavBusy)await refreshItems()}catch(e){if(!poll.notified){toast('连接暂时中断，页面会自动重连：'+e.message,true);poll.notified=true}}finally{pollBusy=false}}
(async()=>{try{await Promise.all([refreshStatus(),refreshJobs(),loadAnnotationDefaults()]);if(jobsCache.length)await selectJob(jobsCache[0].id)}catch(e){toast(e.message,true)}setInterval(poll,4000)})();
