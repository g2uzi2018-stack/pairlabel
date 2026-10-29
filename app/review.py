"""Shared review classification and label-only edits, preserving geometry/evidence."""
import copy,hashlib,json,time
from .classification import assign_class,review_class_id


def agent_classify(item, class_ids, observations, classes, batch, recovery=None):
    """Only relabel canonical geometry; AI review is never human approval."""
    result=copy.deepcopy(item.get('result') or {'objects':[]})
    objects=result.setdefault('objects',[])
    if recovery is not None:
        if objects:raise ValueError('仅允许为空结果采用已有模型轮廓')
        objects.extend(copy.deepcopy(recovery))
    if not objects or len(class_ids)!=len(objects) or len(observations)!=len(objects):
        raise ValueError('类别核对数量必须与已有目标一一对应')
    before_geometry=copy.deepcopy([(o['bbox'],o.get('polygon',[])) for o in objects])
    changes=[]
    for index,(obj,cid,note) in enumerate(zip(objects,class_ids,observations)):
        if cid == -1:cid=None
        old=review_class_id(obj)
        decision_source=note.get('decision_source','luna_visual_review')
        if decision_source not in ('luna_visual_review','primary_reference_adjudication'):raise ValueError('未知类别核对来源')
        assign_class(obj,-1 if cid is None else cid,classes,source='codex_primary_reference' if decision_source=='primary_reference_adjudication' else 'codex_luna')
        if cid is None:obj['classification_reason']=str(note.get('reason') or '看不清／无法可靠分类')[:500]
        obj['classification_review']={'target_key':str(note.get('key',''))[:40],'text':str(note.get('text',''))[:30],
                                      'text_kind':str(note.get('text_kind','observed_or_normalized_glyph'))[:50],
                                      'state':'unreadable' if cid is None else 'readable','model':'gpt-6-luna','reasoning_effort':'max',
                                      'decision_source':decision_source}
        if decision_source=='primary_reference_adjudication':
            obj['classification_review'].update(adjudicator='primary_agent',luna_class_id=note.get('luna_class_id'),reference_key=str(note.get('reference_key',''))[:40])
        if old!=review_class_id(obj):changes.append({'index':index,'before':old,'after':review_class_id(obj)})
    assert before_geometry==[(o['bbox'],o.get('polygon',[])) for o in objects]
    result['classification_review']={'batch':batch,'model':'gpt-6-luna','reasoning_effort':'max','speed':'not_exposed','time':time.time(),
                                     'changed_objects':len(changes),'geometry_preserved':True,'recovered_existing_contour':recovery is not None}
    from .classification import unreadable
    resolved={'class','color_unknown','ambiguous_text','low_confidence','classification_unreadable'}
    if recovery is not None:resolved.update(('empty','missed'))
    prior=item.get('issues')
    if prior is None:prior=result.get('issues',[])
    issues=[copy.deepcopy(issue) for issue in (prior or []) if issue.get('type') not in resolved]
    result['classification_review']['remaining_quality_issues']=len(issues)
    issues += [{'type':'classification_unreadable','message':f'目标 {index+1} 看不清／无法可靠分类'} for index,o in enumerate(objects) if unreadable(o)]
    result['issues']=issues
    return result,issues,changes

# The final item can be 'uncertain' even when a model request failed. Historical
# failed calls from earlier attempts do not classify a subsequently resolved item.
FAILED_SQL = "(status='error' OR (status='uncertain' AND EXISTS (SELECT 1 FROM json_each(items.issues) issue WHERE json_extract(issue.value,'$.type')='inference_error')))"

def review_revision(item):
    state=[item['status'],item['attempt'],item.get('result')]
    return hashlib.sha256(json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def visual_edit(item, edits, classes, width, height, load_run):
    from .geometry import validate_prediction
    if not edits:
        raise ValueError('没有目标可保存，请先画框或采用模型结果；无目标图片请使用高级编辑明确确认')
    if len(edits)>300:raise ValueError('单图目标数量超过 300')
    result=copy.deepcopy(item.get('result') or {'objects':[]})
    original=result.get('objects',[]);objects=[];sources=[];seen=set()
    for edit in edits:
        ref=edit.get('review_ref') or {};kind=ref.get('kind');index=ref.get('index')
        if kind=='result':
            if type(index) is not int or not 0<=index<len(original):raise ValueError('原始目标引用失效，请重新打开图片')
            obj=copy.deepcopy(original[index]);key=('result',index)
        elif kind=='run':
            rid=ref.get('run_id')
            if type(rid) is not int:raise ValueError('模型记录引用无效')
            run=load_run(rid)
            if not run or run['item_id']!=item['id'] or run['status']!='ok':raise ValueError('模型结果不属于当前图片或未成功完成')
            prediction=run['prediction']
            if isinstance(prediction,str):prediction=json.loads(prediction)
            candidates=(prediction or {}).get('objects',[])
            if type(index) is not int or not 0<=index<len(candidates):raise ValueError('模型目标引用无效')
            obj=copy.deepcopy(candidates[index]);key=('run',rid,index)
        elif kind=='drawn':
            obj={'class_id':edit.get('class_id'),'bbox':edit.get('bbox'),'polygon':[],'confidence':0,'text':'','annotation_source':'human_box'}
            key=None
        else:raise ValueError('目标来源无效，请使用画框或模型结果入口')
        if key is not None:
            if key in seen:raise ValueError('同一来源目标被重复添加')
            seen.add(key)
            if edit.get('bbox')!=obj['bbox']:raise ValueError('来源框已变化，请重新打开图片，或重新画框')
        assign_class(obj,edit.get('class_id'),classes)
        # Validate without rebuilding the object: retain canonical source masks and metadata.
        validate_prediction({'objects':[obj]},classes,width,height,normalized=False)
        if kind=='run':obj['annotation_source']='human_adopted_model';obj['human_source_run']=ref['run_id']
        objects.append(obj);sources.append(copy.deepcopy(ref))
    result['objects']=objects;result['human_edited']=True
    reason='用户通过可视化编辑确认目标：'+str(len(objects))+' 个。'
    return result,sources,reason


def recognition_failed(item):
    return item['status'] == 'error' or (item['status'] == 'uncertain' and
        any(issue.get('type') == 'inference_error' for issue in (item.get('issues') or [])))


def relabel(result, class_ids, classes, expected=None):
    result = copy.deepcopy(result or {'objects': []})
    objects = result.get('objects', [])
    if not objects:
        raise ValueError('没有可修改的目标，请重试识别，或用高级编辑补充目标')
    if class_ids is None or len(class_ids) != len(objects):
        raise ValueError('类别数量与目标数量不一致，请刷新后重新确认')
    old_ids = [review_class_id(o) for o in objects]
    if expected is not None and expected != old_ids:
        raise ValueError('类别已被其他操作更新，请重新打开图片后确认')
    by_id = {c['id']: c for c in classes}
    if any(type(cid) is not int or (cid != -1 and cid not in by_id) for cid in class_ids):
        raise ValueError('选择的类别不在本任务类别清单中')
    changes = []
    for index, (obj, cid) in enumerate(zip(objects, class_ids)):
        if cid == review_class_id(obj):
            continue
        changes.append({'index': index, 'before': review_class_id(obj), 'after': cid})
        assign_class(obj,cid,classes)
    if changes:
        result['human_edited'] = True
    reason = '用户确认本图标签（类别未修改）。'
    if changes:
        reason = '用户确认类别修改：' + '；'.join(
            f"目标 {c['index']+1}：{by_id[c['before']]['label'] if c['before']!=-1 else '看不清'} → {by_id[c['after']]['label'] if c['after']!=-1 else '看不清'}" for c in changes)
    return result, changes, reason
