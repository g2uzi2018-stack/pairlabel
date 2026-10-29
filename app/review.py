"""Shared review classification and label-only edits, preserving geometry/evidence."""
import copy,hashlib,json

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
        obj['class_id']=edit.get('class_id')
        # Validate without rebuilding the object: retain canonical source masks and metadata.
        validate_prediction({'objects':[obj]},classes,width,height,normalized=False)
        color=classes[obj['class_id']].get('color')
        if color in ('red','black'):obj['color']=color
        else:obj.pop('color',None)
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
    old_ids = [o['class_id'] for o in objects]
    if expected is not None and expected != old_ids:
        raise ValueError('类别已被其他操作更新，请重新打开图片后确认')
    by_id = {c['id']: c for c in classes}
    if any(type(cid) is not int or cid not in by_id for cid in class_ids):
        raise ValueError('选择的类别不在本任务类别清单中')
    changes = []
    for index, (obj, cid) in enumerate(zip(objects, class_ids)):
        if cid == obj['class_id']:
            continue
        changes.append({'index': index, 'before': obj['class_id'], 'after': cid})
        obj['class_id'] = cid
        if by_id[cid].get('color') in ('red', 'black'):
            obj['color'] = by_id[cid]['color']
        else:
            obj.pop('color', None)
    if changes:
        result['human_edited'] = True
    reason = '用户确认本图标签（类别未修改）。'
    if changes:
        reason = '用户确认类别修改：' + '；'.join(
            f"目标 {c['index']+1}：{by_id[c['before']]['label']} → {by_id[c['after']]['label']}" for c in changes)
    return result, changes, reason
